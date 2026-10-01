import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Dict, Tuple

class DiffHeat2D(nn.Module):
    """
    Differentiable 2D Heat Equation Solver using ADI method.
    Replicates the physics of simulator.py using PyTorch.
    """
    def __init__(self, Lx=2.0, Ly=1.0, nx=100, ny=50, device='cpu'):
        super().__init__()
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.device = device
        
        # Grid coordinates
        x = torch.linspace(0, Lx, nx, device=device)
        y = torch.linspace(0, Ly, ny, device=device)
        self.X, self.Y = torch.meshgrid(x, y, indexing='ij')
        
        # Base matrices (without kappa/dt factors)
        # Lx_base corresponds to D2x operator unscaled
        self.Lx_base = self._build_laplacian_1d_base(nx, self.dx)
        self.Ly_base = self._build_laplacian_1d_base(ny, self.dy)
        
        self.Ix = torch.eye(nx, device=device)
        self.Iy = torch.eye(ny, device=device)
        
        # Operator cache: key = (bc, dt, kappa) -> (Lx_mat, Ly_mat, Ax_inv, Ay_inv, r)
        self._operator_cache = {}

    def _build_laplacian_1d_base(self, n, d):
        """
        Builds the 1D Laplacian diagonal values.
        Returns tridiagonal components or dense matrix.
        We'll use dense for simplicity with torch.linalg.solve (nx=100 is small).
        """
        # Standard central difference: [1, -2, 1]
        # Scaled by 1/d^2 later
        main = -2.0 * torch.ones(n, device=self.device)
        off = torch.ones(n - 1, device=self.device)
        mat = torch.diag(main) + torch.diag(off, diagonal=1) + torch.diag(off, diagonal=-1)
        
        return mat

    def _get_operators(self, bc: str, dt: float, kappa: float):
        """
        Constructs the ADI operators for a specific BC, dt, and kappa.
        """
        # Scale by 1/d^2
        Lx_mat = self.Lx_base / (self.dx**2)
        Ly_mat = self.Ly_base / (self.dy**2)
        
        # Apply BC modifications
        if bc == 'dirichlet':
            # Zero out boundary rows (enforcing U=0 update for residuals, essentially)
            # Actually strictly enforcing 0 value at boundary nodes.
            mask_x = torch.ones_like(Lx_mat)
            mask_x[0, :] = 0; mask_x[-1, :] = 0
            Lx_mat = Lx_mat * mask_x
            
            mask_y = torch.ones_like(Ly_mat)
            mask_y[0, :] = 0; mask_y[-1, :] = 0
            Ly_mat = Ly_mat * mask_y
            
        elif bc == 'neumann':
            # Fix corners for Neumann
            # X-dim
            Lx_mat[0, 0] = -2.0 / (self.dx**2)
            Lx_mat[0, 1] = 2.0 / (self.dx**2)
            Lx_mat[-1, -1] = -2.0 / (self.dx**2)
            Lx_mat[-1, -2] = 2.0 / (self.dx**2)
            
            # Y-dim
            Ly_mat[0, 0] = -2.0 / (self.dy**2)
            Ly_mat[0, 1] = 2.0 / (self.dy**2)
            Ly_mat[-1, -1] = -2.0 / (self.dy**2)
            Ly_mat[-1, -2] = 2.0 / (self.dy**2)
            
        else:
            raise ValueError(f"Unknown BC: {bc}")
            
        # r factor
        r = kappa * dt / 2.0
        
        # Ax = (I - r Lx)
        Ax = self.Ix - r * Lx_mat
        
        # Ay = (I - r Ly)
        Ay = self.Iy - r * Ly_mat
        
        # RHS multipliers (I + r Ly) etc are handled in the loop implicitly 
        # via: U + r * (Ly @ U)
        
        return Lx_mat, Ly_mat, Ax, Ay
    
    def _get_operators_cached(self, bc: str, dt: float, kappa: float):
        """
        Get cached ADI operators with pre-computed inverses.
        Avoids redundant matrix inversions on repeated forward passes.
        """
        # Round dt and kappa to avoid floating-point key issues
        cache_key = (bc, round(dt, 6), round(kappa, 6))
        
        if cache_key not in self._operator_cache:
            Lx_mat, Ly_mat, Ax, Ay = self._get_operators(bc, dt, kappa)
            
            # Pre-compute inverses (expensive, but done only once per configuration)
            Ax_inv = torch.linalg.inv(Ax)
            Ay_inv = torch.linalg.inv(Ay)
            
            r = kappa * dt / 2.0
            
            self._operator_cache[cache_key] = (Lx_mat, Ly_mat, Ax_inv, Ay_inv, r)
        
        return self._operator_cache[cache_key]

    def _deposit_gaussian(self, sources_batch, t_window):
        """
        Creates source field S(x,y) for the batch.
        sources_batch: (B, K, 3) -> x, y, q
        """
        # sources_batch: [Batch, K, 3]
        B, K, _ = sources_batch.shape
        nx, ny = self.nx, self.ny
        
        # S_field: [Batch, nx, ny]
        S_field = torch.zeros(B, nx, ny, device=self.device)
        
        # Precompute grid broadcast
        X_b = self.X.unsqueeze(0) # [1, nx, ny]
        Y_b = self.Y.unsqueeze(0) # [1, nx, ny]
        
        # Auto-sigma
        max_ds = max(self.dx, self.dy)
        sigma = 2.5 * max_ds
        sigma2 = 2.0 * sigma**2
        
        for k in range(K):
            # Extract k-th source parameters
            ks = sources_batch[:, k, :] # [B, 3]
            x0 = ks[:, 0].view(B, 1, 1)
            y0 = ks[:, 1].view(B, 1, 1)
            q  = ks[:, 2].view(B, 1, 1)
            
            # Gaussian
            r_sq = (X_b - x0)**2 + (Y_b - y0)**2
            G = torch.exp(-r_sq / sigma2)
            
            # Integral normalization (sum over grid)
            # We approximate integral as sum * dx * dy
            integral_G = G.sum(dim=(1, 2), keepdim=True) * self.dx * self.dy
            
            # Avoid div by zero
            integral_G = torch.clamp(integral_G, min=1e-8)
            
            # Add to field
            S_field += G * q / integral_G
            
        return S_field

    def bilinear_sample(self, U, sensors_xy, use_half=False):
        """
        Sample U at sensor locations.
        U: [Batch, nx, ny]
        sensors_xy: [Batch, M, 2] (or [M, 2] broadcasted)
        use_half: If True, use float16 for faster sampling (slight precision loss)
        """
        B, nx, ny = U.shape
        if sensors_xy.dim() == 2:
            sensors_xy = sensors_xy.unsqueeze(0).expand(B, -1, -1)
        
        sx = sensors_xy[:, :, 0]
        sy = sensors_xy[:, :, 1]
        
        grid_y = 2.0 * (sx / self.Lx) - 1.0  # SX determines row index (dim 2)
        grid_x = 2.0 * (sy / self.Ly) - 1.0  # SY determines col index (dim 3)
        
        # Stack: (x, y) for grid_sample. 
        grid = torch.stack([grid_x, grid_y], dim=-1) # [B, M, 2]
        grid = grid.unsqueeze(2) # [B, M, 1, 2]
        
        U_in = U.unsqueeze(1) # [B, 1, nx, ny]
        
        # Optimization: Use half precision for sampling if enabled
        if use_half and U.device.type == 'cuda':
            U_in_half = U_in.half()
            grid_half = grid.half()
            sampled = torch.nn.functional.grid_sample(
                U_in_half, grid_half, align_corners=True, mode='bilinear', padding_mode='zeros'
            )
            sampled = sampled.float()  # Convert back to float32
        else:
            sampled = torch.nn.functional.grid_sample(
                U_in, grid, align_corners=True, mode='bilinear', padding_mode='zeros'
            )
        # output: [B, 1, M, 1]
        
        return sampled.squeeze(-1).squeeze(1) # [B, M]

    def forward(self, sources, sensor_xy, T0=0.0, dt=0.004, nt=200, kappa=0.1, bc='dirichlet', use_half_sampling=False):
        """
        Forward simulation.
        sources: [B, K, 3] (x, y, q)
        sensor_xy: [B, M, 2]
        use_half_sampling: If True, use float16 for sensor sampling (slight precision trade-off)
        """
        B = sources.shape[0]
        
        # 1. Get cached operators (avoids redundant matrix inversions)
        Lx_mat, Ly_mat, Ax_inv, Ay_inv, r = self._get_operators_cached(bc, dt, kappa)
        
        # 2. Source Field (Precomputed constant term)
        S_const = self._deposit_gaussian(sources, None) # [B, nx, ny]
        S_step = S_const * (dt / 2.0)
        
        # 3. Initial Condition
        U = torch.full((B, self.nx, self.ny), T0, device=self.device, dtype=sources.dtype)
        
        # Store history
        sensor_readings = []
        
        # Initial reading
        sensor_readings.append(self.bilinear_sample(U, sensor_xy, use_half=use_half_sampling))
        
        t = 0.0
        for n in range(nt):
            # Step 1: Implicit X
            # (I - rLx) U* = (I + rLy) U + S
            term_y = torch.matmul(U, Ly_mat.T) # (Ly U^T)^T effectively
            RHS_x = U + r * term_y + S_step
            
            # Solve Ax U* = RHS_x
            U_star = torch.einsum('ij,bjk->bik', Ax_inv, RHS_x)
            
            # Step 2: Implicit Y
            # (I - rLy) U_next = (I + rLx) U* + S
            term_x = torch.einsum('ij,bjk->bik', Lx_mat, U_star)
            RHS_y = U_star + r * term_x + S_step
            
            # Solve Ay U_next = RHS_y
            U_next = torch.matmul(RHS_y, Ay_inv.T)
            
            U = U_next
            
            # Sample sensors
            sensor_readings.append(self.bilinear_sample(U, sensor_xy, use_half=use_half_sampling))
            
        # Stack time: [Time, B, M] -> [B, Time, M]
        Y_pred = torch.stack(sensor_readings, dim=1)
        
        return Y_pred

if __name__ == '__main__':
    # Simple test
    sim = DiffHeat2D(nx=50, ny=25)
    sources = torch.tensor([[[1.0, 0.5, 1.0]]])
    sensors = torch.tensor([[[0.5, 0.5]]])
    Y = sim(sources, sensors, nt=10)
    print(Y.shape)
    print(Y.requires_grad)