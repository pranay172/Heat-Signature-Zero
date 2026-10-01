
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

from src.diff_simulator import DiffHeat2D

class GDRefiner:
    """
    Gradient Descent Refiner for Heat Source Parameters.
    Uses DiffHeat2D to compute gradients of the forward loss w.r.t source parameters.
    
    Optimizations included:
    - Cosine LR scheduling for better convergence
    - RMSE loss matching competition metric
    - Early stopping with patience
    - Gradient clipping for stability
    - Improved temporal alignment
    """
    def __init__(self, 
                 Lx=2.0, Ly=1.0, 
                 q_range=(0.5, 2.0),
                 nx=100, ny=50,
                 dt=0.004,
                 margin=0.05,
                 device='cpu'):
        self.Lx = Lx
        self.Ly = Ly
        self.q_min, self.q_max = q_range
        self.nx = nx
        self.ny = ny
        self.dt = dt
        self.margin = margin
        self.device = device
        
        # Initialize simulator
        self.sim = DiffHeat2D(Lx, Ly, nx, ny, device=device)
        
        # Speed Optimization: Use torch.compile for faster iteration
        # reduce-overhead mode is best for iterative loops with small tensor ops
        try:
            self.sim = torch.compile(self.sim, mode='reduce-overhead')
            self._compiled = True
        except Exception as e:
            print(f"Warning: torch.compile failed (using eager mode): {e}")
            self._compiled = False



    def refine(self, 
               initial_candidates: List[List[Tuple[float, float, float]]],
               sample: Dict,
               n_steps: int = 50,
               lr: float = 0.1,
               lr_min: float = 0.005,
               patience: int = 8,
               grad_clip: float = 1.0,
               loss_threshold: float = 0.01,
               use_half_sampling: bool = True,
               verbose: bool = False) -> List[List[Tuple[float, float, float]]]:
        """
        Refine a batch of candidates for a SINGLE sample.
        
        Args:
            initial_candidates: List of candidates. Each candidate is a list of (x, y, q).
            sample: The test sample dictionary containing 'Y_noisy', 'sensors_xy', etc.
            n_steps: Maximum number of optimization steps
            lr: Initial learning rate
            lr_min: Minimum learning rate for cosine annealing
            patience: Early stopping patience (stop if no improvement for this many steps)
            grad_clip: Gradient clipping norm
            loss_threshold: Stop early if RMSE drops below this value (good solution found)
            use_half_sampling: Use float16 for sensor sampling (slight speedup on GPU)
            verbose: Print progress
            
        Returns:
            List of refined candidates in same format as input
        """
        
        # 1. Prepare Data
        # Y_target: [Time, M] -> [1, Time, M] -> broadcast to [Batch, Time, M]
        Y_noisy = torch.from_numpy(sample['Y_noisy']).float().to(self.device).unsqueeze(0)
        sensors_xy = torch.from_numpy(sample['sensors_xy']).float().to(self.device) # [M, 2]
        
        meta = sample.get('sample_metadata', {})
        kappa = float(meta.get('kappa', 0.1))
        bc = meta.get('bc', 'dirichlet')
        nt = int(meta.get('nt', Y_noisy.shape[1]))
        
        # 2. Prepare Candidates (Batching)
        n_sources = sample['n_sources']
        batch_size = len(initial_candidates)
        
        # Convert to tensor [Batch, K, 3]
        candidates_array = np.array(initial_candidates, dtype=np.float32)
        params_init = torch.tensor(candidates_array, device=self.device)
        
        # 3. Parameterization to enforce bounds via sigmoid
        def inverse_sigmoid(y, min_val, max_val):
            norm = (y - min_val) / (max_val - min_val)
            norm = torch.clamp(norm, 0.01, 0.99)
            return torch.logit(norm)

        raw_x = inverse_sigmoid(params_init[:, :, 0], 0, self.Lx)
        raw_y = inverse_sigmoid(params_init[:, :, 1], 0, self.Ly)
        raw_q = inverse_sigmoid(params_init[:, :, 2], self.q_min, self.q_max)
        
        # Make leaf tensors
        raw_x = raw_x.clone().detach().requires_grad_(True)
        raw_y = raw_y.clone().detach().requires_grad_(True)
        raw_q = raw_q.clone().detach().requires_grad_(True)
        
        optimizer = torch.optim.Adam([raw_x, raw_y, raw_q], lr=lr)
        
        # Cosine Annealing LR Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_steps, eta_min=lr_min
        )
        
        # Early stopping tracking
        best_loss = float('inf')
        best_state = None
        no_improve_count = 0
        
        # Optimization Loop
        iterator = range(n_steps)
        if verbose:
            iterator = tqdm(iterator, desc="GD Refinement")
            
        for step in iterator:
            optimizer.zero_grad()
            
            # Reparameterize to bounded space
            x = self.Lx * torch.sigmoid(raw_x)
            y = self.Ly * torch.sigmoid(raw_y)
            q = self.q_min + (self.q_max - self.q_min) * torch.sigmoid(raw_q)
            
            # Stack: [Batch, K, 3]
            sources = torch.stack([x, y, q], dim=-1)
            
            # Simulate: Y_pred [Batch, Time, M]
            Y_pred = self.sim(sources, sensors_xy, 
                              T0=float(meta.get('T0', 0.0)),
                              dt=self.dt,
                              nt=nt,
                              kappa=kappa,
                              bc=bc,
                              use_half_sampling=use_half_sampling)
            
            # Temporal alignment using interpolation (better than integer indexing)
            T_encoded = Y_pred.shape[1]
            T_target = Y_noisy.shape[1]
            
            if T_encoded != T_target:
                # Use interpolation for smooth temporal alignment
                # Reshape for interpolate: [B, M, T] for 1D interpolation
                Y_pred_t = Y_pred.transpose(1, 2)  # [B, M, T_encoded]
                Y_pred_t = F.interpolate(Y_pred_t, size=T_target, mode='linear', align_corners=True)
                Y_pred = Y_pred_t.transpose(1, 2)  # [B, T_target, M]
            
            # Compute RMSE loss (matching competition metric)
            diff = Y_pred - Y_noisy
            mse_per_candidate = torch.mean(diff**2, dim=(1, 2))  # [Batch]
            rmse_per_candidate = torch.sqrt(mse_per_candidate + 1e-8)
            loss = rmse_per_candidate.mean()  # Average RMSE across batch
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_([raw_x, raw_y, raw_q], max_norm=grad_clip)
            
            optimizer.step()
            scheduler.step()
            
            # Early stopping check
            current_loss = loss.item()
            if current_loss < best_loss - 1e-6:
                best_loss = current_loss
                best_state = (raw_x.detach().clone(), raw_y.detach().clone(), raw_q.detach().clone())
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Stop if loss is below threshold (good solution found)
            if current_loss < loss_threshold:
                if verbose:
                    print(f"  Loss threshold reached at step {step+1}, loss={current_loss:.6f}")
                break
            
            if no_improve_count >= patience:
                if verbose:
                    print(f"  Early stopping at step {step+1}, best_loss={best_loss:.6f}")
                break
        
        # Restore best state if early stopped
        if best_state is not None:
            raw_x, raw_y, raw_q = best_state
            
        # Final Parameters
        with torch.no_grad():
            x = self.Lx * torch.sigmoid(raw_x)
            y = self.Ly * torch.sigmoid(raw_y)
            q = self.q_min + (self.q_max - self.q_min) * torch.sigmoid(raw_q)
            final_sources = torch.stack([x, y, q], dim=-1)  # [Batch, K, 3]
            
        # Convert back to list format
        refined_candidates = []
        for i in range(batch_size):
            cand = []
            for k in range(n_sources):
                cx = float(final_sources[i, k, 0].item())
                cy = float(final_sources[i, k, 1].item())
                cq = float(final_sources[i, k, 2].item())
                cand.append((cx, cy, cq))
            refined_candidates.append(cand)
            
        return refined_candidates