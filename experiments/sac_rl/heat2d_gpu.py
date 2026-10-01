"""
GPU-accelerated Heat2D solver using PyTorch.
This can be 10-100x faster than scipy for the PDE solve.

OPTIONAL: Only implement if CPU bottleneck persists after increasing n_envs.
"""
import torch
import numpy as np


class Heat2DGPU:
    """GPU-accelerated 2D Heat equation solver using PyTorch."""
    
    def __init__(self, Lx, Ly, nx, ny, kappa, bc="dirichlet", device="cuda"):
        self.Lx, self.Ly = Lx, Ly
        self.nx, self.ny = nx, ny
        self.kappa = kappa
        self.bc = bc
        self.device = device
        
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        
        # Move grid to GPU
        x = torch.linspace(0, Lx, nx, device=device)
        y = torch.linspace(0, Ly, ny, device=device)
        self.X, self.Y = torch.meshgrid(x, y, indexing='ij')
    
    def solve(self, dt, nt, T0=0.0, sources=None, store_every=1):
        """Solve using explicit finite differences (simple but GPU-friendly)."""
        if sources is None:
            sources = []
        
        # Initialize temperature field on GPU
        U = torch.full((self.nx, self.ny), T0, device=self.device, dtype=torch.float32)
        
        times, Us = [], []
        t = 0.0
        
        # Precompute Fourier number
        r_x = self.kappa * dt / (self.dx ** 2)
        r_y = self.kappa * dt / (self.dy ** 2)
        
        # Stability check
        if r_x + r_y > 0.5:
            raise ValueError(f"Unstable! r_x + r_y = {r_x + r_y} > 0.5. Reduce dt.")
        
        for n in range(nt):
            if n % store_every == 0:
                times.append(t)
                Us.append(U.cpu().numpy().copy())
            
            # Source term
            S = self._source_field(t, sources)
            
            # Explicit update (vectorized on GPU)
            U_new = U.clone()
            U_new[1:-1, 1:-1] = (
                U[1:-1, 1:-1]
                + r_x * (U[2:, 1:-1] - 2*U[1:-1, 1:-1] + U[:-2, 1:-1])
                + r_y * (U[1:-1, 2:] - 2*U[1:-1, 1:-1] + U[1:-1, :-2])
                + dt * S[1:-1, 1:-1]
            )
            
            # Boundary conditions
            if self.bc == "dirichlet":
                U_new[0, :] = 0
                U_new[-1, :] = 0
                U_new[:, 0] = 0
                U_new[:, -1] = 0
            
            U = U_new
            t += dt
        
        # Store final state
        times.append(t)
        Us.append(U.cpu().numpy().copy())
        
        return np.array(times), np.array(Us)
    
    def _source_field(self, t, sources):
        """Compute source field as Gaussian."""
        S = torch.zeros((self.nx, self.ny), device=self.device)
        
        for src in sources:
            t0, t1 = src.get("on", (0, 1e9))
            if not (t0 <= t <= t1):
                continue
            
            Q = src.get("q", 0.0)
            if Q == 0:
                continue
            
            x0, y0 = src["x"], src["y"]
            sigma = src.get("sigma", 2.5 * max(self.dx, self.dy))
            
            r_sq = (self.X - x0)**2 + (self.Y - y0)**2
            G = torch.exp(-r_sq / (2 * sigma**2))
            
            # Normalize
            integral = torch.sum(G) * self.dx * self.dy
            if integral > 0:
                S += G * Q / integral
        
        return S
    
    def sample_sensors(self, U_field, sensor_xy):
        """Bilinear interpolation at sensor positions."""
        U_tensor = torch.from_numpy(U_field).to(self.device)
        sensors_tensor = torch.from_numpy(sensor_xy).to(self.device)
        
        out = []
        for sx, sy in sensors_tensor:
            fx = sx / self.dx
            fy = sy / self.dy
            ix = int(torch.clamp(torch.floor(fx), 0, self.nx - 2))
            iy = int(torch.clamp(torch.floor(fy), 0, self.ny - 2))
            tx = fx - ix
            ty = fy - iy
            
            v = (
                (1 - tx) * (1 - ty) * U_tensor[ix, iy]
                + tx * (1 - ty) * U_tensor[ix + 1, iy]
                + (1 - tx) * ty * U_tensor[ix, iy + 1]
                + tx * ty * U_tensor[ix + 1, iy + 1]
            )
            out.append(v.item())
        
        return np.array(out)


# Example usage:
if __name__ == "__main__":
    solver = Heat2DGPU(2.0, 1.0, 50, 25, 0.1, device="cuda")
    sources = [{"x": 1.0, "y": 0.5, "q": 1.5}]
    
    import time
    start = time.time()
    times, fields = solver.solve(dt=0.004, nt=1000, sources=sources)
    print(f"GPU solve: {time.time() - start:.3f}s")
