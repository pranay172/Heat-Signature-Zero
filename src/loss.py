"""
Loss functions for heat source prediction.

Implements Hungarian matching for permutation-invariant loss computation.
"""

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import Tuple, Optional
import numpy as np


def hungarian_matching(
    pred: torch.Tensor,
    target: torch.Tensor,
    source_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute optimal assignment between predictions and targets using Hungarian algorithm.
    
    Args:
        pred: (B, K_max, 3) predicted sources
        target: (B, K_max, 3) ground truth sources
        source_mask: (B, K_max) valid source mask
    
    Returns:
        row_indices: (B, K_max) indices for predictions
        col_indices: (B, K_max) indices for targets
    """
    B = pred.size(0)
    K_max = pred.size(1)
    device = pred.device
    
    row_indices = torch.zeros(B, K_max, dtype=torch.long, device=device)
    col_indices = torch.zeros(B, K_max, dtype=torch.long, device=device)
    
    for b in range(B):
        # Get number of valid sources
        K_valid = source_mask[b].sum().item()
        
        if K_valid == 0:
            continue
        
        # Extract valid predictions and targets
        pred_b = pred[b, :K_valid]  # (K_valid, 3)
        target_b = target[b, :K_valid]  # (K_valid, 3)
        
        # Compute pairwise cost matrix (L2 distance)
        # Convert to float32 for cdist compatibility with AMP (half precision not supported)
        cost = torch.cdist(pred_b.float(), target_b.float(), p=2)  # (K_valid, K_valid)
        
        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
        
        row_indices[b, :K_valid] = torch.tensor(row_ind, device=device)
        col_indices[b, :K_valid] = torch.tensor(col_ind, device=device)
    
    return row_indices, col_indices


def hungarian_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    source_mask: torch.Tensor,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Permutation-invariant MSE loss using Hungarian matching.
    
    Args:
        pred: (B, K_max, 3) predicted sources (normalized)
        target: (B, K_max, 3) ground truth sources (normalized)
        source_mask: (B, K_max) valid source mask
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        Scalar loss (if reduction != 'none')
    """
    B = pred.size(0)
    device = pred.device
    
    # Get optimal matching
    row_indices, col_indices = hungarian_matching(pred, target, source_mask)
    
    total_loss = torch.tensor(0.0, device=device)
    n_valid = 0
    
    for b in range(B):
        K_valid = source_mask[b].sum().item()
        
        if K_valid == 0:
            continue
        
        # Get matched predictions and targets
        row_idx = row_indices[b, :K_valid]
        col_idx = col_indices[b, :K_valid]
        
        matched_pred = pred[b, row_idx]  # (K_valid, 3)
        matched_target = target[b, col_idx]  # (K_valid, 3)
        
        # MSE loss
        loss = F.mse_loss(matched_pred, matched_target, reduction='sum')
        total_loss = total_loss + loss
        n_valid += K_valid
    
    if reduction == 'mean':
        return total_loss / max(n_valid, 1)
    elif reduction == 'sum':
        return total_loss
    else:
        return total_loss


def hungarian_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    source_mask: torch.Tensor,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Permutation-invariant L1 loss using Hungarian matching.
    
    Args:
        pred: (B, K_max, 3) predicted sources (normalized)
        target: (B, K_max, 3) ground truth sources (normalized)
        source_mask: (B, K_max) valid source mask
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        Scalar loss
    """
    B = pred.size(0)
    device = pred.device
    
    row_indices, col_indices = hungarian_matching(pred, target, source_mask)
    
    total_loss = torch.tensor(0.0, device=device)
    n_valid = 0
    
    for b in range(B):
        K_valid = source_mask[b].sum().item()
        
        if K_valid == 0:
            continue
        
        row_idx = row_indices[b, :K_valid]
        col_idx = col_indices[b, :K_valid]
        
        matched_pred = pred[b, row_idx]
        matched_target = target[b, col_idx]
        
        loss = F.l1_loss(matched_pred, matched_target, reduction='sum')
        total_loss = total_loss + loss
        n_valid += K_valid
    
    if reduction == 'mean':
        return total_loss / max(n_valid, 1)
    elif reduction == 'sum':
        return total_loss
    else:
        return total_loss


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    source_mask: torch.Tensor,
    mse_weight: float = 1.0,
    l1_weight: float = 0.5,
) -> Tuple[torch.Tensor, dict]:
    """
    Combined MSE + L1 loss with Hungarian matching.
    
    Args:
        pred: (B, K_max, 3) predicted sources
        target: (B, K_max, 3) ground truth sources
        source_mask: (B, K_max) valid source mask
        mse_weight: Weight for MSE loss
        l1_weight: Weight for L1 loss
    
    Returns:
        total_loss: Weighted sum of losses
        loss_dict: Dictionary with individual loss values
    """
    mse = hungarian_mse_loss(pred, target, source_mask)
    l1 = hungarian_l1_loss(pred, target, source_mask)
    
    total = mse_weight * mse + l1_weight * l1
    
    return total, {
        'loss': total.item(),
        'mse_loss': mse.item(),
        'l1_loss': l1.item(),
    }


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    source_mask: torch.Tensor,
    Lx: float = 2.0,
    Ly: float = 1.0,
    q_min: float = 0.5,
    q_max: float = 2.0,
) -> dict:
    """
    Compute evaluation metrics after Hungarian matching.
    
    Args:
        pred: (B, K_max, 3) predicted sources (normalized)
        target: (B, K_max, 3) ground truth sources (normalized)
        source_mask: (B, K_max) valid source mask
        Lx, Ly: Domain dimensions
        q_min, q_max: Intensity range
    
    Returns:
        Dictionary with metrics:
        - mae_x, mae_y, mae_q: Mean absolute error for each parameter
        - mae_position: MAE for position (Euclidean distance)
        - rmse: Root mean square error
    """
    B = pred.size(0)
    
    row_indices, col_indices = hungarian_matching(pred, target, source_mask)
    
    all_errors_x = []
    all_errors_y = []
    all_errors_q = []
    all_distances = []
    
    for b in range(B):
        K_valid = source_mask[b].sum().item()
        
        if K_valid == 0:
            continue
        
        row_idx = row_indices[b, :K_valid]
        col_idx = col_indices[b, :K_valid]
        
        matched_pred = pred[b, row_idx]  # (K_valid, 3)
        matched_target = target[b, col_idx]  # (K_valid, 3)
        
        # Denormalize for interpretable metrics
        pred_x = matched_pred[:, 0] * Lx
        pred_y = matched_pred[:, 1] * Ly
        pred_q = matched_pred[:, 2] * (q_max - q_min) + q_min
        
        target_x = matched_target[:, 0] * Lx
        target_y = matched_target[:, 1] * Ly
        target_q = matched_target[:, 2] * (q_max - q_min) + q_min
        
        # Compute errors
        all_errors_x.extend((pred_x - target_x).abs().tolist())
        all_errors_y.extend((pred_y - target_y).abs().tolist())
        all_errors_q.extend((pred_q - target_q).abs().tolist())
        
        # Position distance
        distances = torch.sqrt((pred_x - target_x)**2 + (pred_y - target_y)**2)
        all_distances.extend(distances.tolist())
    
    if len(all_errors_x) == 0:
        return {'mae_x': 0, 'mae_y': 0, 'mae_q': 0, 'mae_position': 0, 'rmse': 0}
    
    return {
        'mae_x': np.mean(all_errors_x),
        'mae_y': np.mean(all_errors_y),
        'mae_q': np.mean(all_errors_q),
        'mae_position': np.mean(all_distances),
        'rmse': np.sqrt(np.mean(np.array(all_errors_x)**2 + 
                                np.array(all_errors_y)**2 + 
                                np.array(all_errors_q)**2)),
    }


# ==============================================================================
# Test
# ==============================================================================
if __name__ == '__main__':
    print("Testing loss functions...")
    
    # Create dummy data
    B, K_max = 4, 3
    pred = torch.rand(B, K_max, 3)
    target = torch.rand(B, K_max, 3)
    source_mask = torch.zeros(B, K_max, dtype=torch.bool)
    source_mask[0, 0] = True  # 1 source
    source_mask[1, :2] = True  # 2 sources
    source_mask[2, 0] = True  # 1 source
    source_mask[3, :2] = True  # 2 sources
    
    print(f"Source mask:\n{source_mask}")
    
    # Test Hungarian matching
    row_idx, col_idx = hungarian_matching(pred, target, source_mask)
    print(f"\nRow indices: {row_idx}")
    print(f"Col indices: {col_idx}")
    
    # Test losses
    mse_loss = hungarian_mse_loss(pred, target, source_mask)
    l1_loss = hungarian_l1_loss(pred, target, source_mask)
    print(f"\nMSE loss: {mse_loss.item():.4f}")
    print(f"L1 loss: {l1_loss.item():.4f}")
    
    # Test combined loss
    total, loss_dict = combined_loss(pred, target, source_mask)
    print(f"\nCombined loss: {total.item():.4f}")
    print(f"Loss dict: {loss_dict}")
    
    # Test metrics
    metrics = compute_metrics(pred, target, source_mask)
    print(f"\nMetrics: {metrics}")
    
    # Test backprop
    pred.requires_grad = True
    loss = hungarian_mse_loss(pred, target, source_mask)
    loss.backward()
    print(f"\nGradient computed: {pred.grad is not None}")
    print(f"Gradient shape: {pred.grad.shape}")
    
    print("\n✅ All loss tests passed!")
