"""
Integration code to add to your train.py

Replace the constraints-related code in train.py with this.
"""

import torch
from fuzzy_penalties import (
    compute_fuzzy_constraints_loss,
    compute_violation_metrics
)

# ============================================================================
# ADD THIS TO imports in train.py
# ============================================================================
# from fuzzy_penalties import compute_fuzzy_constraints_loss, compute_violation_metrics


# ============================================================================
# REPLACE run_train_phase function with this:
# ============================================================================

def run_train_phase(model, P, Z, logger, epoch, phase):
    """
    Run one training phase with fuzzy logic constraints support.
    
    Supports both Łukasiewicz and Product t-norms.
    """
    assert phase == "train"
    Z['current_epoch'] = epoch
    
    # Accumulator for violation metrics
    violation_accum = {
        'impl_viol_rate': 0.0,
        'excl_viol_rate': 0.0,
        'mean_impl_penalty': 0.0,
        'mean_excl_penalty': 0.0,
        'n_batches': 0
    }
    
    model.train()
    
    for batch in Z["dataloaders"][phase]:
        # Move data to GPU
        batch["image"] = batch["image"].to(Z["device"], non_blocking=True)
        batch["labels_np"] = batch["label_vec_obs"].clone().numpy()
        batch["label_vec_obs"] = batch["label_vec_obs"].to(Z["device"], non_blocking=True)
        
        # Forward pass
        Z["optimizer"].zero_grad()
        with torch.set_grad_enabled(True):
            batch["logits"] = model.f(batch["image"])
            batch["preds"] = torch.sigmoid(batch["logits"])
            
            if batch["preds"].dim() == 1:
                batch["preds"] = torch.unsqueeze(batch["preds"], 0)
            
            batch["label_vec_est"] = model.g(batch["idx"])
            batch["preds_np"] = batch["preds"].clone().detach().cpu().numpy()
            
            # ---- Compute violation metrics (for logging only, no grad) ----
            if Z.get("use_constraints", False):
                with torch.no_grad():
                    viol_metrics = compute_violation_metrics(
                        batch["preds"],
                        Z.get("implication_rules", []),
                        Z.get("exclusion_rules", []),
                        threshold=0.5
                    )
                    # Accumulate
                    violation_accum['impl_viol_rate'] += viol_metrics['impl_viol_rate']
                    violation_accum['excl_viol_rate'] += viol_metrics['excl_viol_rate']
                    
                    # Select penalty based on fuzzy type
                    fuzzy_type = Z.get("fuzzy_type", "lukasiewicz")
                    if fuzzy_type == "lukasiewicz":
                        violation_accum['mean_impl_penalty'] += viol_metrics['mean_impl_penalty_luk']
                        violation_accum['mean_excl_penalty'] += viol_metrics['mean_excl_penalty_luk']
                    else:  # product
                        violation_accum['mean_impl_penalty'] += viol_metrics['mean_impl_penalty_prod']
                        violation_accum['mean_excl_penalty'] += viol_metrics['mean_excl_penalty_prod']
                    
                    violation_accum['n_batches'] += 1
            
            # ---- Constraints warmup schedule ----
            if Z.get("use_constraints", False):
                base_lambda_c = Z.get("lambda_constraints", 0.0)
                warm = Z.get("constraints_warmup_epochs", 0)
                
                if warm > 0 and epoch < warm:
                    lambda_c = base_lambda_c * float(epoch + 1) / float(warm)
                else:
                    lambda_c = base_lambda_c
            else:
                lambda_c = 0.0
            
            # ---- Compute fuzzy logic constraints loss ----
            if Z.get("use_constraints", False) and lambda_c > 0:
                fuzzy_type = Z.get("fuzzy_type", "lukasiewicz")
                
                loss_constraints = compute_fuzzy_constraints_loss(
                    preds=batch["preds"],
                    implication_pairs=Z.get("implication_rules", []),
                    exclusion_pairs=Z.get("exclusion_rules", []),
                    fuzzy_type=fuzzy_type,
                    lambda_impl=Z.get("lambda_impl", 1.0),
                    lambda_excl=Z.get("lambda_excl", 1.0),
                    impl_weights=Z.get("impl_weights", None),
                    excl_weights=Z.get("excl_weights", None),
                    reduction="mean"
                )
            else:
                loss_constraints = 0.0
            
            # ---- Compute main loss (ROLE or other) ----
            batch = compute_batch_loss(batch, P, Z)
            
            # ---- Add constraints to total loss ----
            if Z.get("use_constraints", False):
                batch["loss_tensor"] = batch["loss_tensor"] + (lambda_c * loss_constraints)
                batch["constraint_loss_np"] = float(loss_constraints) if torch.is_tensor(loss_constraints) else loss_constraints
            else:
                batch["constraint_loss_np"] = 0.0
        
        # Backward pass
        batch["loss_tensor"].backward()
        Z["optimizer"].step()
        
        # Save current batch data
        logger.update_phase_data(batch)
        
        # Optional: increment global step counter
        Z["global_step"] = Z.get("global_step", 0) + 1
    
    # ---- Average violation metrics over epoch ----
    if violation_accum['n_batches'] > 0:
        n = violation_accum['n_batches']
        epoch_violations = {
            'impl_viol_rate': violation_accum['impl_viol_rate'] / n,
            'excl_viol_rate': violation_accum['excl_viol_rate'] / n,
            'mean_impl_penalty': violation_accum['mean_impl_penalty'] / n,
            'mean_excl_penalty': violation_accum['mean_excl_penalty'] / n,
        }
    else:
        epoch_violations = {
            'impl_viol_rate': 0.0,
            'excl_viol_rate': 0.0,
            'mean_impl_penalty': 0.0,
            'mean_excl_penalty': 0.0,
        }
    
    return epoch_violations


# ============================================================================
# REPLACE run_eval_phase function with this:
# ============================================================================

def run_eval_phase(model, P, Z, logger, epoch, phase):
    """
    Run one evaluation phase (val/test) with violation tracking.
    """
    assert phase in ["val", "test"]
    
    # Accumulator for violation metrics
    violation_accum = {
        'impl_viol_rate': 0.0,
        'excl_viol_rate': 0.0,
        'mean_impl_penalty': 0.0,
        'mean_excl_penalty': 0.0,
        'n_batches': 0
    }
    
    model.eval()
    
    for batch in Z["dataloaders"][phase]:
        # Move data to GPU
        batch["image"] = batch["image"].to(Z["device"], non_blocking=True)
        batch["labels_np"] = batch["label_vec_obs"].clone().numpy()
        batch["label_vec_obs"] = batch["label_vec_obs"].to(Z["device"], non_blocking=True)
        
        # Forward pass
        with torch.set_grad_enabled(False):
            batch["logits"] = model.f(batch["image"])
            batch["preds"] = torch.sigmoid(batch["logits"])
            
            if batch["preds"].dim() == 1:
                batch["preds"] = torch.unsqueeze(batch["preds"], 0)
            
            batch["preds_np"] = batch["preds"].clone().detach().cpu().numpy()
            batch["loss_np"] = -1
            batch["reg_loss_np"] = -1
            
            # Compute violation metrics
            if Z.get("use_constraints", False):
                viol_metrics = compute_violation_metrics(
                    batch["preds"],
                    Z.get("implication_rules", []),
                    Z.get("exclusion_rules", []),
                    threshold=0.5
                )
                violation_accum['impl_viol_rate'] += viol_metrics['impl_viol_rate']
                violation_accum['excl_viol_rate'] += viol_metrics['excl_viol_rate']
                
                fuzzy_type = Z.get("fuzzy_type", "lukasiewicz")
                if fuzzy_type == "lukasiewicz":
                    violation_accum['mean_impl_penalty'] += viol_metrics['mean_impl_penalty_luk']
                    violation_accum['mean_excl_penalty'] += viol_metrics['mean_excl_penalty_luk']
                else:
                    violation_accum['mean_impl_penalty'] += viol_metrics['mean_impl_penalty_prod']
                    violation_accum['mean_excl_penalty'] += viol_metrics['mean_excl_penalty_prod']
                
                violation_accum['n_batches'] += 1
        
        # Save current batch data
        logger.update_phase_data(batch)
    
    # Average violations
    if violation_accum['n_batches'] > 0:
        n = violation_accum['n_batches']
        epoch_violations = {
            'impl_viol_rate': violation_accum['impl_viol_rate'] / n,
            'excl_viol_rate': violation_accum['excl_viol_rate'] / n,
            'mean_impl_penalty': violation_accum['mean_impl_penalty'] / n,
            'mean_excl_penalty': violation_accum['mean_excl_penalty'] / n,
        }
    else:
        epoch_violations = {
            'impl_viol_rate': 0.0,
            'excl_viol_rate': 0.0,
            'mean_impl_penalty': 0.0,
            'mean_excl_penalty': 0.0,
        }
    
    return epoch_violations


# ============================================================================
# UPDATE train function - add W&B logging:
# ============================================================================

def train(model, P, Z):
    """
    Train the model with enhanced W&B logging.
    """
    best_weights_f = copy.deepcopy(model.f.state_dict())
    best_weights_g = copy.deepcopy(model.g.state_dict())
    logger = train_logger(P)
    
    for epoch in range(P["num_epochs"]):
        print("Epoch {}/{}".format(epoch, P["num_epochs"] - 1))
        
        for phase in ["train", "val", "test"]:
            # Reset phase metrics
            logger.reset_phase_data()
            t_init = time.time()
            
            # Run phase
            if phase == "train":
                phase_violations = run_train_phase(model, P, Z, logger, epoch, phase)
            else:
                phase_violations = run_eval_phase(model, P, Z, logger, epoch, phase)
            
            # Compute phase metrics
            logger.compute_phase_metrics(phase, epoch, model.g.get_estimated_labels())
            
            # Print epoch status
            logger.report(t_init, time.time(), phase, epoch)
            
            # ============================================================
            # W&B LOGGING - COMPREHENSIVE
            # ============================================================
            try:
                import wandb
                
                # Get stop metric (mAP)
                stop_metric_name = P.get("stop_metric", "map")
                try:
                    variant = P.get("val_set_variant", None) if phase in ["val", "test"] else None
                    metric_val = logger.get_stop_metric(phase, epoch, variant)
                except:
                    metric_val = None
                
                # Build logging dict
                log_dict = {"epoch": epoch}
                
                # Main metric
                if metric_val is not None:
                    log_dict[f"{phase}/{stop_metric_name}"] = float(metric_val)
                
                # Violation metrics
                if phase_violations is not None:
                    log_dict[f"{phase}/impl_viol_rate"] = phase_violations['impl_viol_rate']
                    log_dict[f"{phase}/excl_viol_rate"] = phase_violations['excl_viol_rate']
                    log_dict[f"{phase}/mean_impl_penalty"] = phase_violations['mean_impl_penalty']
                    log_dict[f"{phase}/mean_excl_penalty"] = phase_violations['mean_excl_penalty']
                    log_dict[f"{phase}/total_penalty"] = (
                        phase_violations['mean_impl_penalty'] + 
                        phase_violations['mean_excl_penalty']
                    )
                
                # Constraint warmup (only for train)
                if phase == "train" and Z.get("use_constraints", False):
                    base_lambda = Z.get("lambda_constraints", 0.0)
                    warmup = Z.get("constraints_warmup_epochs", 0)
                    if warmup > 0 and epoch < warmup:
                        current_lambda = base_lambda * (epoch + 1) / warmup
                    else:
                        current_lambda = base_lambda
                    log_dict["train/effective_lambda_constraints"] = current_lambda
                
                # Log to W&B
                wandb.log(log_dict, commit=(phase == "test"))
                
            except Exception as e:
                print(f"Warning: W&B logging failed: {e}")
            
            # Update best epoch
            new_best = False
            try:
                new_best = logger.update_best_results(phase, epoch, P.get("val_set_variant"))
            except:
                try:
                    new_best = logger.update_best_results(phase, epoch)
                except:
                    pass
            
            if new_best:
                print("*** new best weights ***")
                best_weights_f = copy.deepcopy(model.f.state_dict())
                best_weights_g = copy.deepcopy(model.g.state_dict())
                
                # Save and upload to W&B
                try:
                    import wandb
                    os.makedirs(P["save_path"], exist_ok=True)
                    f_path = os.path.join(P["save_path"], f"best_model_state_f_epoch{epoch}.pt")
                    g_path = os.path.join(P["save_path"], f"best_model_state_g_epoch{epoch}.pt")
                    torch.save(best_weights_f, f_path)
                    torch.save(best_weights_g, g_path)
                    
                    artifact = wandb.Artifact(
                        f"{P['experiment_name']}_best_epoch_{epoch}", 
                        type="model"
                    )
                    artifact.add_file(f_path)
                    artifact.add_file(g_path)
                    wandb.log_artifact(artifact)
                except:
                    pass
    
    print("")
    print("*** TRAINING COMPLETE ***")
    print("Best epoch: {}".format(logger.best_epoch))
    
    try:
        print("Best epoch validation score: {:.2f}".format(
            logger.get_stop_metric("val", logger.best_epoch, P.get("val_set_variant"))
        ))
    except:
        pass
    
    try:
        print("Best epoch test score: {:.2f}".format(
            logger.get_stop_metric("test", logger.best_epoch, "clean")
        ))
    except:
        pass
    
    return P, model, logger, best_weights_f, best_weights_g


# ============================================================================
# UPDATE initialize_training_run - add fuzzy config:
# ============================================================================

def initialize_training_run(P, feature_extractor, linear_classifier, estimated_labels):
    """
    Set up for model training with fuzzy logic support.
    """
    os.makedirs(P["save_path"], exist_ok=True)
    np.random.seed(P["seed"])
    
    Z = {}
    
    # Device
    Z["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(Z["device"]))
    
    # Data
    Z["datasets"] = datasets.get_data(P)
    observed_label_matrix = Z["datasets"]["train"].label_matrix_obs
    P["num_classes"] = Z["datasets"]["train"].num_classes
    
    # Dataloaders
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
    
    # ---- Fuzzy logic constraints configuration ----
    Z["use_constraints"] = P.get("use_constraints", False)
    Z["fuzzy_type"] = P.get("fuzzy_type", "lukasiewicz")  # or "product"
    
    Z["implication_rules"] = P.get("implication_rules", [])
    Z["exclusion_rules"] = P.get("exclusion_rules", [])
    Z["impl_weights"] = P.get("impl_weights", None)
    Z["excl_weights"] = P.get("excl_weights", None)
    
    Z["lambda_constraints"] = float(P.get("lambda_constraints", 0.0))
    Z["lambda_impl"] = float(P.get("lambda_impl", 1.0))
    Z["lambda_excl"] = float(P.get("lambda_excl", 1.0))
    Z["constraints_warmup_epochs"] = int(P.get("constraints_warmup_epochs", 0))
    
    print(f"\nConstraints Configuration:")
    print(f"  Use constraints: {Z['use_constraints']}")
    if Z['use_constraints']:
        print(f"  Fuzzy type: {Z['fuzzy_type']}")
        print(f"  λ_constraints: {Z['lambda_constraints']}")
        print(f"  λ_impl: {Z['lambda_impl']}")
        print(f"  λ_excl: {Z['lambda_excl']}")
        print(f"  Warmup epochs: {Z['constraints_warmup_epochs']}")
        print(f"  Implication rules: {len(Z['implication_rules'])}")
        print(f"  Exclusion rules: {len(Z['exclusion_rules'])}")
    
    # Model
    model = models.MultilabelModel(
        P, feature_extractor, linear_classifier, observed_label_matrix, estimated_labels
    )
    
    # Optimization
    f_params = [param for param in list(model.f.parameters()) if param.requires_grad]
    g_params = [param for param in list(model.g.parameters()) if param.requires_grad]
    opt_params = [
        {"params": f_params, "lr": P["lr"]},
        {"params": g_params, "lr": P["lr_mult"] * P["lr"]},
    ]
    Z["optimizer"] = torch.optim.Adam(opt_params, lr=P["lr"])
    
    return P, Z, model