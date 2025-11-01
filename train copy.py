import os
import copy
import time
import json
import numpy as np
import torch
import datasets
import models
from losses import compute_batch_loss
import datetime
from instrumentation import train_logger
import pickle

import torch
from ccn.constraints_group import ConstraintsGroup
from ccn.clauses_group import ClausesGroup
from ccn.constraints_layer import ConstraintsLayer

def _new_accum():
    return dict(impl_pen=0.0, excl_pen=0.0, impl_viol=0.0, excl_viol=0.0, n=0)

def _accum_update(acc, mean_impl, mean_excl, rate_impl, rate_excl):
    # each arg can be tensor() or float
    to_f = lambda x: float(x.detach().cpu().item()) if torch.is_tensor(x) else float(x)
    acc["impl_pen"]  += to_f(mean_impl)
    acc["excl_pen"]  += to_f(mean_excl)
    acc["impl_viol"] += to_f(rate_impl)
    acc["excl_viol"] += to_f(rate_excl)
    acc["n"] += 1

def _accum_finalize(acc):
    if acc["n"] == 0:
        return dict(mean_impl_pen=np.nan, mean_excl_pen=np.nan, viol_impl=np.nan, viol_excl=np.nan)
    n = acc["n"]
    return dict(
        mean_impl_pen = acc["impl_pen"]/n,
        mean_excl_pen = acc["excl_pen"]/n,
        viol_impl     = acc["impl_viol"]/n,
        viol_excl     = acc["excl_viol"]/n,
    )


@torch.no_grad()
def _penalties_mean(preds, impl_pairs, excl_pairs):
    # means of the *raw* per-pair penalties (not thresholded)
    mean_impl = fuzzy_implication_penalty(preds, impl_pairs, reduction="mean").detach()
    mean_excl = fuzzy_exclusion_penalty(preds, excl_pairs, reduction="mean").detach()
    return mean_impl, mean_excl

@torch.no_grad()
def _violation_rates(preds, impl_pairs, excl_pairs, thr=0.5):
    """
    Counts violations on hardened preds: 
      implication violates when pA>=thr and pB<thr
      exclusion   violates when pC>=thr and pD>=thr
    Returns mean rates over batch.
    """
    if preds.numel() == 0:
        return preds.new_zeros(()), preds.new_zeros(())
    B, C = preds.shape
    hard = (preds >= thr).to(torch.float32)

    viol_impl = []
    for (a, b) in impl_pairs or []:
        viol_impl.append((hard[:, a] > 0.5) & (hard[:, b] < 0.5))
    viol_excl = []
    for (c, d) in excl_pairs or []:
        viol_excl.append((hard[:, c] > 0.5) & (hard[:, d] > 0.5))

    if len(viol_impl) == 0:
        rate_impl = preds.new_zeros(())
    else:
        rate_impl = torch.stack([v.float().mean() for v in viol_impl]).mean()

    if len(viol_excl) == 0:
        rate_excl = preds.new_zeros(())
    else:
        rate_excl = torch.stack([v.float().mean() for v in viol_excl]).mean()

    return rate_impl, rate_excl


# ===== Logic penalties (Łukasiewicz) – training only =====
import torch.nn.functional as F

def _relu(x):
    return F.relu(x)

@torch.no_grad()
def _ensure_rule_tensors_device(rules, device):
    if not rules:
        return None, None
    a = torch.tensor([i for i, _ in rules], device=device, dtype=torch.long)
    b = torch.tensor([j for _, j in rules], device=device, dtype=torch.long)
    return a, b

def fuzzy_implication_penalty(preds: torch.Tensor,
                              implication_pairs,
                              weights=None,
                              eps: float = 1e-6,
                              reduction: str = "mean") -> torch.Tensor:
    """
    Implication A -> B with Łukasiewicz fuzzy logic:
      truth = min(1, 1 - pA + pB)  =>  penalty = ReLU(pA - pB)
    preds: (B, C) in [0,1]
    implication_pairs: list[(A_idx, B_idx)]
    """
    if not implication_pairs:
        return preds.new_zeros(())
    device = preds.device
    A_idx, B_idx = _ensure_rule_tensors_device(implication_pairs, device)
    pA = preds[:, A_idx].clamp(eps, 1 - eps)
    pB = preds[:, B_idx].clamp(eps, 1 - eps)
    pen = _relu(pA - pB)  # (B, R)
    if weights is not None:
        w = torch.as_tensor(weights, device=device, dtype=preds.dtype).view(1, -1)
        pen = pen * w
    if reduction == "mean":
        return pen.mean()
    elif reduction == "sum":
        return pen.sum()
    return pen

def fuzzy_exclusion_penalty(preds: torch.Tensor,
                            exclusion_pairs,
                            weights=None,
                            eps: float = 1e-6,
                            reduction: str = "mean") -> torch.Tensor:
    """
    Mutual exclusion C ⟂ D with Łukasiewicz t-norm:
      penalty = ReLU(pC + pD - 1)
    preds: (B, C) in [0,1]
    exclusion_pairs: list[(C_idx, D_idx)]
    """
    if not exclusion_pairs:
        return preds.new_zeros(())
    device = preds.device
    C_idx, D_idx = _ensure_rule_tensors_device(exclusion_pairs, device)
    pC = preds[:, C_idx].clamp(eps, 1 - eps)
    pD = preds[:, D_idx].clamp(eps, 1 - eps)
    pen = _relu(pC + pD - 1.0)  # (B, R)
    if weights is not None:
        w = torch.as_tensor(weights, device=device, dtype=preds.dtype).view(1, -1)
        pen = pen * w
    if reduction == "mean":
        return pen.mean()
    elif reduction == "sum":
        return pen.sum()
    return pen

def constraints_loss(preds: torch.Tensor,
                     implication_pairs,
                     exclusion_pairs,
                     lambda_impl: float = 1.0,
                     lambda_excl: float = 1.0,
                     impl_weights=None,
                     excl_weights=None,
                     reduction: str = "mean") -> torch.Tensor:
    """
    L_constraints = lambda_impl * L_impl + lambda_excl * L_excl
    """
    L_impl = fuzzy_implication_penalty(preds, implication_pairs, impl_weights, reduction=reduction)
    L_excl = fuzzy_exclusion_penalty(preds, exclusion_pairs, excl_weights, reduction=reduction)
    return lambda_impl * L_impl + lambda_excl * L_excl
# ===== End of logic penalties =====

import wandb  # ====== W&B already imported in your file ======

def init_constraints_layer(rules_path, num_classes, device):
    group = ConstraintsGroup(rules_path)
    clauses = ClausesGroup.from_constraints_group(group)
    layer = ConstraintsLayer.from_clauses_group(
        clauses, num_classes=num_classes, centrality="katz"
    ).to(device)
    return group, layer


def run_train_phase(model, P, Z, logger, epoch, phase):
    """
    Run one training phase.
    """
    assert phase == "train"
    Z['current_epoch'] = epoch
    accum = _new_accum()
    model.train()
    for batch in Z["dataloaders"][phase]:
        # move data to GPU:
        batch["image"] = batch["image"].to(Z["device"], non_blocking=True)
        batch["labels_np"] = batch["label_vec_obs"].clone().numpy()
        batch["label_vec_obs"] = batch["label_vec_obs"].to(Z["device"], non_blocking=True)
        # forward pass:
        Z["optimizer"].zero_grad()
        with torch.set_grad_enabled(True):
            batch["logits"] = model.f(batch["image"])
            batch["preds"] = torch.sigmoid(batch["logits"])
            mean_impl, mean_excl = _penalties_mean(
                batch["preds"], Z["implication_rules"], Z["exclusion_rules"]
            )
            rate_impl, rate_excl = _violation_rates(
                batch["preds"], Z["implication_rules"], Z["exclusion_rules"], thr=0.5
            )
            _accum_update(accum, mean_impl, mean_excl, rate_impl, rate_excl)

            if batch["preds"].dim() == 1:
                batch["preds"] = torch.unsqueeze(batch["preds"], 0)
            batch["label_vec_est"] = model.g(batch["idx"])
            batch["preds_np"] = batch["preds"].clone().detach().cpu().numpy()
                        # ---- Constraints warmup schedule (optional but recommended)
            base_lambda_c = Z["lambda_constraints"]
            warm = Z["constraints_warmup_epochs"]
            if warm > 0 and epoch < warm:
                lambda_c = base_lambda_c * float(epoch + 1) / float(warm)
            else:
                lambda_c = base_lambda_c

            # ---- Compute fuzzy logic penalties from probabilities (no projection)
            loss_cons = constraints_loss(
                preds=batch["preds"],                             # (B, C) probs
                implication_pairs=Z["implication_rules"],
                exclusion_pairs=Z["exclusion_rules"],
                lambda_impl=Z["lambda_impl"],
                lambda_excl=Z["lambda_excl"],
                impl_weights=Z["impl_weights"],
                excl_weights=Z["excl_weights"],
                reduction="mean",
            )

            batch = compute_batch_loss(batch, P, Z)
            if Z["use_constraints"]:
                batch["loss_tensor"] = batch["loss_tensor"] + (lambda_c * loss_cons)
        # backward pass:
        batch["loss_tensor"].backward()
        Z["optimizer"].step()
        # save current batch data:
        logger.update_phase_data(batch)

        # optional: increment a global step counter if you want epoch+step in logs
        Z["global_step"] = Z.get("global_step", 0) + 1


    return _accum_finalize(accum)


def run_eval_phase(model, P, Z, logger, epoch, phase):
    """
    Run one evaluation phase.
    """
    assert phase in ["val", "test"]
    model.eval()
    accum = _new_accum()
    for batch in Z["dataloaders"][phase]:
        # move data to GPU:
        batch["image"] = batch["image"].to(Z["device"], non_blocking=True)
        batch["labels_np"] = batch["label_vec_obs"].clone().numpy()
        batch["label_vec_obs"] = batch["label_vec_obs"].to(Z["device"], non_blocking=True)
        # forward pass:
        with torch.set_grad_enabled(False):
            batch["logits"] = model.f(batch["image"])
            batch["preds"] = torch.sigmoid(batch["logits"])
            if batch["preds"].dim() == 1:
                batch["preds"] = torch.unsqueeze(batch["preds"], 0)
            batch["preds_np"] = batch["preds"].clone().detach().cpu().numpy()
            batch["loss_np"] = -1
            batch["reg_loss_np"] = -1

            mean_impl, mean_excl = _penalties_mean(
                batch["preds"], Z["implication_rules"], Z["exclusion_rules"]
            )
            rate_impl, rate_excl = _violation_rates(
                batch["preds"], Z["implication_rules"], Z["exclusion_rules"], thr=0.5
            )
            _accum_update(accum, mean_impl, mean_excl, rate_impl, rate_excl)
        # save current batch data:
        logger.update_phase_data(batch)

    return _accum_finalize(accum)


def train(model, P, Z):
    """
    Train the model.
    """
    best_weights_f = copy.deepcopy(model.f.state_dict())
    best_weights_g = copy.deepcopy(model.g.state_dict())
    logger = train_logger(P)  # initialize logger

    for epoch in range(P["num_epochs"]):
        print("Epoch {}/{}".format(epoch, P["num_epochs"] - 1))

        for phase in ["train", "val", "test"]:
            # reset phase metrics:
            logger.reset_phase_data()
            t_init = time.time()

            if phase == "train":
                phase_logic = run_train_phase(model, P, Z, logger, epoch, phase)
            else:
                phase_logic = run_eval_phase(model, P, Z, logger, epoch, phase)

            # save end-of-phase metrics:
            logger.compute_phase_metrics(phase, epoch, model.g.get_estimated_labels())

            # print epoch status:
            logger.report(t_init, time.time(), phase, epoch)

            try:
                if phase_logic is not None:
                    wandb.log({
                        "epoch": epoch,
                        f"{phase}/mean_impl_pen_epoch": phase_logic["mean_impl_pen"],
                        f"{phase}/mean_excl_pen_epoch": phase_logic["mean_excl_pen"],
                        f"{phase}/viol_impl_epoch":     phase_logic["viol_impl"],
                        f"{phase}/viol_excl_epoch":     phase_logic["viol_excl"],
                    }, commit=False)
            except Exception:
                pass

            try:
                # Attempt to log the stop metric (e.g., map) for this phase
                stop_metric_name = P.get("stop_metric", None)
                if stop_metric_name is not None:
                    # For val/test we include val_set_variant; for train it might not apply
                    variant = P.get("val_set_variant", None)
                    try:
                        metric_val = logger.get_stop_metric(phase, epoch, variant)
                    except Exception:
                        # fallback: try calling without variant
                        metric_val = logger.get_stop_metric(phase, epoch, None)
                    if metric_val is not None:
                        wandb.log({"epoch": epoch, f"{phase}/{stop_metric_name}": float(metric_val)}, commit=True)
                else:
                    # fallback: try to dump any accessible summary metrics from logger
                    # many loggers provide a dict; attempt to access logger.phase_metrics
                    phase_metrics = getattr(logger, "phase_metrics", None)
                    if isinstance(phase_metrics, dict):
                        log_dict = {"epoch": epoch}
                        for k, v in phase_metrics.items():
                            log_dict[f"{phase}/{k}"] = float(v)
                        wandb.log(log_dict, commit=True)
            except Exception:
                pass

            # update best epoch, if applicable:
            new_best = False
            try:
                new_best = logger.update_best_results(phase, epoch, P["val_set_variant"])
            except Exception:
                # if logger signature differs, still try to update best via returned info
                try:
                    new_best = logger.update_best_results(phase, epoch)
                except Exception:
                    new_best = False

            if new_best:
                print("*** new best weights ***")
                best_weights_f = copy.deepcopy(model.f.state_dict())
                best_weights_g = copy.deepcopy(model.g.state_dict())
                # === W&B ADDED: save & upload best weights as artifact ===
                try:
                    # ensure save path exists
                    os.makedirs(P["save_path"], exist_ok=True)
                    f_path = os.path.join(P["save_path"], f"best_model_state_f_epoch{epoch}.pt")
                    g_path = os.path.join(P["save_path"], f"best_model_state_g_epoch{epoch}.pt")
                    torch.save(best_weights_f, f_path)
                    torch.save(best_weights_g, g_path)
                    artifact = wandb.Artifact(f"{P['experiment_name']}_best_epoch_{epoch}", type="model")
                    artifact.add_file(f_path)
                    artifact.add_file(g_path)
                    wandb.log_artifact(artifact)
                except Exception:
                    pass

    print("")
    print("*** TRAINING COMPLETE ***")
    print("Best epoch: {}".format(logger.best_epoch))
    try:
        print(
            "Best epoch validation score: {:.2f}".format(
                logger.get_stop_metric("val", logger.best_epoch, P["val_set_variant"])
            )
        )
    except Exception:
        pass
    try:
        print(
            "Best epoch test score:       {:.2f}".format(
                logger.get_stop_metric("test", logger.best_epoch, "clean")
            )
        )
    except Exception:
        pass

    return P, model, logger, best_weights_f, best_weights_g


def initialize_training_run(P, feature_extractor, linear_classifier, estimated_labels):
    """
    Set up for model training.
    """
    os.makedirs(P["save_path"], exist_ok=True)
    np.random.seed(P["seed"])

    Z = {}

    # constraints:
    Z["use_constraints"] = P.get("use_constraints", False)

    # accelerator:
    Z["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(Z["device"]))

    # data:
    Z["datasets"] = datasets.get_data(P)

    # observed label matrix:
    observed_label_matrix = Z["datasets"]["train"].label_matrix_obs

    # save dataset-specific parameters:
    P["num_classes"] = Z["datasets"]["train"].num_classes

    # dataloaders:
    Z["dataloaders"] = {}
    for phase in ["train", "val", "test"]:
        Z["dataloaders"][phase] = torch.utils.data.DataLoader(
            Z["datasets"][phase],
            batch_size=P["bsize"],
            shuffle=phase == "train",
            sampler=None,
            num_workers=P["num_workers"],
            drop_last=True,
        )

    # group, layer = init_constraints_layer("coco-rules.txt", P["num_classes"], Z["device"])
    # Z["constraints"] = layer

        # ---- (A) If you already have explicit pairs in P, use them:
    Z["implication_rules"] = P.get("implication_rules", [])
    Z["exclusion_rules"]   = P.get("exclusion_rules", [])
    Z["impl_weights"]      = P.get("impl_weights", None)   # list of len |implication_rules|
    Z["excl_weights"]      = P.get("excl_weights", None)   # list of len |exclusion_rules|

    # Global constraint weight + warmup
    Z["lambda_constraints"]        = float(P.get("lambda_constraints", 0.0))
    Z["lambda_impl"]               = float(P.get("lambda_impl", 1.0))
    Z["lambda_excl"]               = float(P.get("lambda_excl", 1.0))
    Z["constraints_warmup_epochs"] = int(P.get("constraints_warmup_epochs", 0))


    # model:
    model = models.MultilabelModel(
        P, feature_extractor, linear_classifier, observed_label_matrix, estimated_labels
    )

    # optimization objects:
    f_params = [param for param in list(model.f.parameters()) if param.requires_grad]
    g_params = [param for param in list(model.g.parameters()) if param.requires_grad]
    opt_params = [
        {"params": f_params, "lr": P["lr"]},
        {"params": g_params, "lr": P["lr_mult"] * P["lr"]},
    ]
    Z["optimizer"] = torch.optim.Adam(opt_params, lr=P["lr"])

    return P, Z, model


def execute_training_run(
    P, feature_extractor, linear_classifier, estimated_labels=None
):
    """
    Initialize, run the training process, and save the results.
    """
    P, Z, model = initialize_training_run(
        P, feature_extractor, linear_classifier, estimated_labels
    )
    model.to(Z["device"])

    # === W&B ADDED: watch the model so W&B can track gradients/parameters ===
    try:
        wandb.watch(model, log="all", log_freq=100)
    except Exception:
        pass

    P, model, logger, best_weights_f, best_weights_g = train(model, P, Z)

    print(
        "\nSaving best weights for f to {}/best_model_state_f.pt".format(P["save_path"])
    )
    torch.save(best_weights_f, os.path.join(P["save_path"], "best_model_state_f.pt"))
    print(
        "\nSaving best weights for g to {}/best_model_state_g.pt".format(P["save_path"])
    )
    torch.save(best_weights_g, os.path.join(P["save_path"], "best_model_state_g.pt"))

    final_logs = logger.get_logs()
    print("\nSaving session data to {}/logs.json".format(P["save_path"]))
    with open(os.path.join(P["save_path"], "logs.json"), "w") as f:
        json.dump(final_logs, f)

    print("\nSaving session data to {}/params.json".format(P["save_path"]))
    with open(os.path.join(P["save_path"], "params.json"), "w") as f:
        json.dump(P, f)

    print("\nReverting model to best weights.")
    model.f.load_state_dict(best_weights_f)
    model.g.load_state_dict(best_weights_g)

    return (
        model.f.feature_extractor,
        model.f.linear_classifier,
        model.g.get_estimated_labels(),
        final_logs,
    )


if __name__ == "__main__":

    lookup = {
        "feat_dim": {"resnet50": 2048},
        "expected_num_pos": {"pascal": 1.5, "coco": 2.9, "nuswide": 1.9, "cub": 31.4},
        "linear_init_params": {
            "an_ls": {
                "pascal": {"linear_init_lr": 1e-4, "linear_init_bsize": 8},
                "coco": {"linear_init_lr": 1e-4, "linear_init_bsize": 8},
                "nuswide": {"linear_init_lr": 1e-4, "linear_init_bsize": 16},
                "cub": {"linear_init_lr": 1e-4, "linear_init_bsize": 8},
            },
            "role": {
                "pascal": {"linear_init_lr": 1e-3, "linear_init_bsize": 16},
                "coco": {"linear_init_lr": 1e-3, "linear_init_bsize": 16},
                "nuswide": {"linear_init_lr": 1e-3, "linear_init_bsize": 16},
                "cub": {"linear_init_lr": 1e-3, "linear_init_bsize": 8},
            },
        },
    }

    P = {}

    # Constraints:
    P["use_constraints"] = True

    # Top-level parameters:
    P["dataset"] = "pascal"  # pascal, coco, nuswide, cub
    P["loss"] = "role"  # bce, bce_ls, iun, iu, pr, an, an_ls, wan, epr, role
    P["train_mode"] = "linear_init"  # linear_fixed_features, end_to_end, linear_init
    P["val_set_variant"] = "clean"  # clean, observed

    # Paths and filenames:
    P["experiment_name"] = "multi_label_experiment"
    P["load_path"] = "./data"
    P["save_path"] = "./results"

    # Optimization parameters:
    if P["train_mode"] == "linear_init":
        P["linear_init_lr"] = lookup["linear_init_params"][P["loss"]][P["dataset"]]["linear_init_lr"]
        P["linear_init_bsize"] = lookup["linear_init_params"][P["loss"]][P["dataset"]]["linear_init_bsize"]
    P["lr_mult"] = 10.0
    P["stop_metric"] = "map"

    # Loss-specific parameters:
    P["ls_coef"] = 0.1

    # Additional parameters:
    P["seed"] = 1200
    P["use_pretrained"] = True
    P["num_workers"] = 0

    # Dataset parameters:
    P["split_seed"] = 1200
    P["val_frac"] = 0.2
    P["ss_seed"] = 999
    P["ss_frac_train"] = 1.0
    P["ss_frac_val"] = 1.0

    # Dependent parameters:
    if P["loss"] in ["bce", "bce_ls"]:
        P["train_set_variant"] = "clean"
    else:
        P["train_set_variant"] = "observed"
    if P["train_mode"] == "end_to_end":
        P["num_epochs"] = 10
        P["freeze_feature_extractor"] = False
        P["use_feats"] = False
        P["arch"] = "resnet50"
    elif P["train_mode"] == "linear_init":
        P["num_epochs"] = 25
        P["freeze_feature_extractor"] = True
        P["use_feats"] = True
        P["arch"] = "linear"
    elif P["train_mode"] == "linear_fixed_features":
        P["num_epochs"] = 25
        P["freeze_feature_extractor"] = True
        P["use_feats"] = True
        P["arch"] = "linear"
    else:
        raise NotImplementedError("Unknown training mode.")
    P["feature_extractor_arch"] = "resnet50"
    P["feat_dim"] = lookup["feat_dim"][P["feature_extractor_arch"]]
    P["expected_num_pos"] = lookup["expected_num_pos"][P["dataset"]]
    P["train_feats_file"] = "./data/{}/train_features_imagenet_{}.npy".format(P["dataset"], P["feature_extractor_arch"])
    P["val_feats_file"] = "./data/{}/val_features_imagenet_{}.npy".format(P["dataset"], P["feature_extractor_arch"])
    P["lambda_constraints"] = 0.5          # global multiplier for constraints block
    P["lambda_impl"] = 1.0                 # relative weight for implication
    P["lambda_excl"] = 1.0                 # relative weight for exclusion
    P["constraints_warmup_epochs"] = 3     # linearly ramp λ_c from 0 → λ_c over first N epochs

    # Provide rules either as explicit index pairs…
    P["implication_rules"] = [(0, 20), (1, 20), (3, 20), (5, 20), (6, 20), (13, 20), (18, 20), (2, 21), (7, 21), (9, 21), (11, 21), (12, 21), (16, 21), (4, 22), (8, 22), (10, 22), (15, 22), (17, 22), (19, 22)]

    # P["exclusion_rules"]   = [(day_idx, night_idx)]

    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="ibrahimkaliljh-student",
        # Set the wandb project where this run will be logged.
        project="spmll_thesis_logical_constraints",
        # Track hyperparameters and run metadata.
        config=P,
    )
    # === W&B ADDED: ensure config is updated (safer if edit after init) ===
    try:
        wandb.config.update(P)
    except Exception:
        pass

    # run training process:
    best_params = None
    best_lr = None
    best_bsize = None
    best_val_score = -np.inf
    best_test_score = None
    now_str = datetime.datetime.now().strftime("%Y_%m_%d_%X").replace(":", "-")
    if P["train_mode"] == "linear_init":
        print("training linear classifier with fixed hyperparameters:")
        print("- linear_init_lr: {}".format(P["linear_init_lr"]))
        print("- linear_init_bsize: {}".format(P["linear_init_bsize"]))
        P["bsize"] = P["linear_init_bsize"]
        P["lr"] = P["linear_init_lr"]
        P["save_path"] = "./results/" + P["experiment_name"] + "_" + now_str + "_" + P["dataset"]
        os.makedirs(P["save_path"], exist_ok=False)
        P_temp = copy.deepcopy(P)  # re-set hyperparameter dict
        # after linear init:
        (
            feature_extractor_init,
            linear_classifier_init,
            estimated_labels_init,
            logs,
        ) = execute_training_run(P_temp, feature_extractor=None, linear_classifier=None)
        print("saving objects")
        save_obj = (feature_extractor_init, linear_classifier_init, estimated_labels_init, logs)
        with open("linear_init/linear_init_pascal.pkl", "wb") as f:
            pickle.dump(save_obj, f)
        print("fine-tuning from trained linear classifier")

    print("Results without constraints:")
    # === W&B ADDED: Attempt to log final summary to W&B ===
    try:
        wandb.log({"final/best_val_score": best_val_score, "final/best_test_score": best_test_score})
    except Exception:
        pass

    try:
        print("best run: {}".format(best_params["save_path"]))
        print("- learning rate: {}".format(best_params["lr"]))
        print("- batch size:    {}".format(best_params["bsize"]))
        print("- val score:     {}".format(best_val_score))
        print("- test score:    {}".format(best_test_score))
    except Exception:
        # if best_params is None, just skip
        pass

    # === W&B ADDED: finish run cleanly ===
    try:
        wandb.finish()
    except Exception:
        pass
