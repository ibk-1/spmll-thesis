import torch
import torch.nn.functional as F

LOG_EPSILON = 1e-5

'''
helper functions
'''

def neg_log(x):
    return - torch.log(x + LOG_EPSILON)

def log_loss(preds, targs):
    return targs * neg_log(preds)

def expected_positive_regularizer(preds, expected_num_pos, norm='2'):
    # Assumes predictions in [0,1].
    if norm == '1':
        reg = torch.abs(preds.sum(1).mean(0) - expected_num_pos)
    elif norm == '2':
        reg = (preds.sum(1).mean(0) - expected_num_pos)**2
    else:
        raise NotImplementedError
    return reg
    
    
# Instead of: F.mse_loss(preds, preds_hat.detach())

def safe_consistency_loss(preds_raw, preds_constrained, threshold=0.05):
    # Only learn from significant constraint corrections
    mask = (torch.abs(preds_raw - preds_constrained) > threshold).float()
    kl_loss = preds_constrained * torch.log(preds_constrained / (preds_raw + LOG_EPSILON) + LOG_EPSILON)
    return (kl_loss * mask).sum() / mask.sum().clamp(min=1.0)

# # In your loss function:
# consistency_loss = safe_consistency_loss(preds, preds_hat.detach()) + \
#                   safe_consistency_loss(estimated_labels, est_hat.detach())

'''
loss functions
'''

def loss_bce(batch, P, Z):
    # unpack:
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']
    # input validation: 
    assert not torch.any(observed_labels == -1)
    assert P['train_set_variant'] == 'clean'
    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = neg_log(1.0 - preds[observed_labels == 0])
    reg_loss = None
    return loss_mtx, reg_loss

def loss_bce_ls(batch, P, Z):
    # unpack:
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']
    # input validation: 
    assert not torch.any(observed_labels == -1)
    assert P['train_set_variant'] == 'clean'
    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = (1.0 - P['ls_coef']) * neg_log(preds[observed_labels == 1]) + P['ls_coef'] * neg_log(1.0 - preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = (1.0 - P['ls_coef']) * neg_log(1.0 - preds[observed_labels == 0]) + P['ls_coef'] * neg_log(preds[observed_labels == 0])
    reg_loss = None
    return loss_mtx, reg_loss

def loss_iun(batch, P, Z):
    # unpack:
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']
    true_labels = batch['label_vec_true']
    # input validation: 
    assert torch.min(observed_labels) >= 0
    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx[true_labels == -1] = neg_log(1.0 - preds[true_labels == -1]) # This loss gets unrealistic access to true negatives.
    reg_loss = None
    return loss_mtx, reg_loss

def loss_iu(batch, P, Z):
    # unpack:
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']
    # input validation: 
    assert torch.any(observed_labels == 1) # must have at least one observed positive
    assert torch.any(observed_labels == -1) # must have at least one observed negative
    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx[observed_labels == -1] = neg_log(1.0 - preds[observed_labels == -1])
    reg_loss = None
    return loss_mtx, reg_loss

def loss_pr(batch, P, Z):
    # unpack:
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']
    batch_size = int(batch['label_vec_obs'].size(0))
    num_classes = int(batch['label_vec_obs'].size(1))
    # input validation: 
    assert torch.min(observed_labels) >= 0
    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels)
    for n in range(batch_size):
        preds_neg = preds[n, :][observed_labels[n, :] == 0]
        for i in range(num_classes):
            if observed_labels[n, i] == 1:
                torch.nonzero(observed_labels[n, :])
                loss_mtx[n, i] = torch.sum(torch.clamp(1.0 - preds[n, i] + preds_neg, min=0))
    reg_loss = None
    return loss_mtx, reg_loss

def loss_an(batch, P, Z):
    # unpack:
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']
    # input validation: 
    assert torch.min(observed_labels) >= 0
    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = neg_log(1.0 - preds[observed_labels == 0])
    reg_loss = None
    return loss_mtx, reg_loss

def loss_an_ls(batch, P, Z):
    # unpack:
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']
    # input validation: 
    assert torch.min(observed_labels) >= 0
    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = (1.0 - P['ls_coef']) * neg_log(preds[observed_labels == 1]) + P['ls_coef'] * neg_log(1.0 - preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = (1.0 - P['ls_coef']) * neg_log(1.0 - preds[observed_labels == 0]) + P['ls_coef'] * neg_log(preds[observed_labels == 0])
    reg_loss = None
    return loss_mtx, reg_loss

def loss_wan(batch, P, Z):
    # unpack:
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']
    # input validation: 
    assert torch.min(observed_labels) >= 0
    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = neg_log(1.0 - preds[observed_labels == 0]) / float(P['num_classes'] - 1)
    reg_loss = None
    
    return loss_mtx, reg_loss

def loss_epr(batch, P, Z):
    # unpack:
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']
    # input validation:
    assert torch.min(observed_labels) >= 0
    # compute loss w.r.t. observed positives:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    # compute regularizer: 
    reg_loss = expected_positive_regularizer(preds, P['expected_num_pos'], norm='2') / (P['num_classes'] ** 2)
    return loss_mtx, reg_loss

def loss_role(batch, P, Z):
    # unpack:
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']
    estimated_labels = batch['label_vec_est']
    # input validation:
    assert torch.min(observed_labels) >= 0
    # (image classifier) compute loss w.r.t. observed positives:
    loss_mtx_pos_1 = torch.zeros_like(observed_labels)
    loss_mtx_pos_1[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    # (image classifier) compute loss w.r.t. label estimator outputs:
    estimated_labels_detached = estimated_labels.detach()
    loss_mtx_cross_1 = estimated_labels_detached * neg_log(preds) + (1.0 - estimated_labels_detached) * neg_log(1.0 - preds)
    # (image classifier) compute regularizer: 
    reg_1 = expected_positive_regularizer(preds, P['expected_num_pos'], norm='2') / (P['num_classes'] ** 2)
    # (label estimator) compute loss w.r.t. observed positives:
    loss_mtx_pos_2 = torch.zeros_like(observed_labels)
    loss_mtx_pos_2[observed_labels == 1] = neg_log(estimated_labels[observed_labels == 1])
    # (label estimator) compute loss w.r.t. image classifier outputs:
    preds_detached = preds.detach()
    loss_mtx_cross_2 = preds_detached * neg_log(estimated_labels) + (1.0 - preds_detached) * neg_log(1.0 - estimated_labels)
    # (label estimator) compute regularizer:
    reg_2 = expected_positive_regularizer(estimated_labels, P['expected_num_pos'], norm='2') / (P['num_classes'] ** 2)
    # compute final loss matrix:
    reg_loss = 0.5 * (reg_1 + reg_2)
    loss_mtx = 0.5 * (loss_mtx_pos_1 + loss_mtx_pos_2)
    loss_mtx += 0.5 * (loss_mtx_cross_1 + loss_mtx_cross_2)
    
    return loss_mtx, reg_loss




def loss_role_logics_improved(batch, P, Z):
    # Your existing code up to regularizers...
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']
    estimated_labels = batch['label_vec_est']
    
    layer = Z['constraints']
    preds_hat = layer(preds, iterative=True)
    est_hat = layer(estimated_labels, iterative=True)
    
    # Positive label losses (unchanged)
    loss_mtx_pos_1 = torch.zeros_like(observed_labels)
    loss_mtx_pos_1[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    
    loss_mtx_pos_2 = torch.zeros_like(observed_labels)
    loss_mtx_pos_2[observed_labels == 1] = neg_log(estimated_labels[observed_labels == 1])
    
    # Cross-learning with constrained targets (your approach - good!)
    est_hat_detached = est_hat.detach()
    loss_mtx_cross_1 = est_hat_detached * neg_log(preds) + (1.0 - est_hat_detached) * neg_log(1.0 - preds)
    
    preds_hat_detached = preds_hat.detach()
    loss_mtx_cross_2 = preds_hat_detached * neg_log(estimated_labels) + (1.0 - preds_hat_detached) * neg_log(1.0 - estimated_labels)
    
    # FIX: Use raw predictions for regularizers
    reg_1 = expected_positive_regularizer(preds, P['expected_num_pos'], norm='2') / (P['num_classes'] ** 2)
    reg_2 = expected_positive_regularizer(estimated_labels, P['expected_num_pos'], norm='2') / (P['num_classes'] ** 2)
    
    # ADD: Gentle consistency loss to prevent over-correction
    epoch = Z.get('current_epoch', 0.01)
    
    # Start with very gentle constraints
    base_consistency_weight = P.get('consistency_coef', 0.3)  # Much smaller than usual
    
    # Curriculum learning: gradually increase constraint influence
    curriculum_factor = min(1.0, epoch / 10.0)
    
    # Calculate consistency losses
    consistency_loss_f = F.mse_loss(preds, preds_hat.detach()) 
    consistency_loss_g = F.mse_loss(estimated_labels, est_hat.detach())
    
    total_consistency = base_consistency_weight * curriculum_factor * (consistency_loss_f + consistency_loss_g)
    
    # Combine losses
    reg_loss = 0.5 * (reg_1 + reg_2) + total_consistency
    loss_mtx = 0.5 * (loss_mtx_pos_1 + loss_mtx_pos_2) + 0.5 * (loss_mtx_cross_1 + loss_mtx_cross_2)
    
    return loss_mtx, reg_loss

# It's good practice to define a numerically stable neg_log helper
def neg_log(x, eps=1e-8):
    """Calculates the negative logarithm of a tensor, clamping values to avoid log(0)."""
    return -torch.log(x.clamp(min=eps))

def loss_role_logics_optimized(batch, P, Z):
    """
    An optimized version of the loss function focusing on vectorization,
    numerical stability, and clarity.
    """
    # --- 1. Unpack Tensors ---
    preds = batch['preds']
    observed_labels = batch['label_vec_obs'] # Assumed to be a binary mask (0s and 1s)
    estimated_labels = batch['label_vec_est']
    
    # --- 2. Apply Constraints ---
    # This part remains the same as it depends on the custom layer logic.
    layer = Z['constraints']
    preds_hat = layer(preds, iterative=True)
    est_hat = layer(estimated_labels, iterative=True)

    # --- 3. Calculate Loss Components ---
    
    # Positive label losses (vectorized for performance)
    # Instead of creating a zero tensor and indexing, we multiply by the binary mask directly.
    # This is significantly faster, especially on a GPU.
    loss_pos_1 = neg_log(preds) * observed_labels
    loss_pos_2 = neg_log(estimated_labels) * observed_labels
    
    # Cross-learning losses (using built-in BCE for stability and speed)
    # F.binary_cross_entropy is more numerically stable than the manual formula.
    # 'reduction="none"' ensures it returns a loss matrix of the same shape.
    loss_cross_1 = F.binary_cross_entropy(preds, est_hat.detach(), reduction='none')
    loss_cross_2 = F.binary_cross_entropy(estimated_labels, preds_hat.detach(), reduction='none')

    # --- 4. Calculate Regularization and Consistency ---

    # Regularization loss (code is fine, slightly consolidated for readability)
    reg_1 = expected_positive_regularizer(preds, P['expected_num_pos'], norm='2')
    reg_2 = expected_positive_regularizer(estimated_labels, P['expected_num_pos'], norm='2')
    reg_loss_base = (reg_1 + reg_2) / (2 * (P['num_classes'] ** 2))

    # Consistency loss (with curriculum learning)
    epoch = Z.get('current_epoch', 0.0)
    base_consistency_weight = P.get('consistency_coef', 0.3)
    curriculum_factor = min(1.0, epoch / 10.0) # Simple and clear
    
    # Using detach() directly in the loss function call
    consistency_loss = F.mse_loss(preds, preds_hat.detach()) + F.mse_loss(estimated_labels, est_hat.detach())
    total_consistency = base_consistency_weight * curriculum_factor * consistency_loss
    
    # --- 5. Combine and Return ---
    
    # Final regularization loss
    reg_loss = reg_loss_base + total_consistency
    
    # Final main loss matrix
    loss_mtx = 0.5 * (loss_pos_1 + loss_pos_2 + loss_cross_1 + loss_cross_2)
    
    return loss_mtx, reg_loss




loss_functions = {
    'bce': loss_bce,
    'bce_ls': loss_bce_ls,
    'iun': loss_iun,
    'iu': loss_iu,
    'pr': loss_pr,
    'an': loss_an,
    'an_ls': loss_an_ls,
    'wan': loss_wan,
    'epr': loss_epr,
    'role': loss_role_logics_optimized,
    'role_logics': loss_role_logics_optimized
}

'''
top-level wrapper
'''

def compute_batch_loss(batch, P, Z):
    
    assert batch['preds'].dim() == 2
    
    batch_size = int(batch['preds'].size(0))
    num_classes = int(batch['preds'].size(1))
    
    loss_denom_mtx = (num_classes * batch_size) * torch.ones_like(batch['preds'])
    
    # input validation:
    assert torch.max(batch['label_vec_obs']) <= 1
    assert torch.min(batch['label_vec_obs']) >= -1
    assert batch['preds'].size() == batch['label_vec_obs'].size()
    assert P['loss'] in loss_functions
    
    # validate predictions:
    assert torch.max(batch['preds']) <= 1
    assert torch.min(batch['preds']) >= 0
    
    # compute loss for each image and class:
    loss_mtx, reg_loss = loss_functions[P['loss']](batch, P, Z)
    main_loss = (loss_mtx / loss_denom_mtx).sum()
    
    if reg_loss is not None:
        batch['loss_tensor'] = main_loss + reg_loss
        batch['reg_loss_np'] = reg_loss.clone().detach().cpu().numpy()
    else:
        batch['loss_tensor'] = main_loss
        batch['reg_loss_np'] = 0.0
    batch['loss_np'] = batch['loss_tensor'].clone().detach().cpu().numpy()
    
    return batch
