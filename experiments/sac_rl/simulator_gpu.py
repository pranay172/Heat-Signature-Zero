import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Tuple, Union


class Heat2DGPU:
    """
    GPU-accelerated 2D heat equation solver using PyTorch.
    Supports batch processing for efficient RL training.
    Uses ADI (Alternating Direction Implicit) method with PyTorch operations.
    """

    def __init__(
        self,
        Lx: float,
        Ly: float,
        nx: int,
        ny: int,
        kappa: float,
        bc: str = "dirichlet",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize the GPU-accelerated heat solver.
        
        Args:
            Lx, Ly: Domain dimensions
            nx, ny: Grid resolution
            kappa: Thermal diffusivity
            bc: Boundary condition ("dirichlet" or "neumann")
            device: PyTorch device ('cuda', 'cpu', or None for auto-detect)
            dtype: Data type for tensors
        """
        self.Lx, self.Ly = Lx, Ly
        self.nx, self.ny = nx, ny
        self.kappa = kappa
        
        supported_bc = ["dirichlet", "neumann"]
        if bc not in supported_bc:
            print(f"Warning: BC '{bc}' not supported. Defaulting to 'dirichlet'.")
            bc = "dirichlet"
        self.bc = bc
        
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.dtype = dtype
        
        # Grid parameters
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        
        # Create coordinate grids on GPU
        x = torch.linspace(0, Lx, nx, device=self.device, dtype=dtype)
        y = torch.linspace(0, Ly, ny, device=self.device, dtype=dtype)
        self.X, self.Y = torch.meshgrid(x, y, indexing='ij')
        
        # Precompute Laplacian operators for ADI
        self._build_adi_matrices()
        
    def _build_adi_matrices(self):
        """Build tridiagonal matrices for ADI method."""
        # X-direction tridiagonal matrix
        self.Lx_diag = torch.full((self.nx,), -2.0 / self.dx**2, 
                                   device=self.device, dtype=self.dtype)
        self.Lx_off = torch.full((self.nx - 1,), 1.0 / self.dx**2, 
                                  device=self.device, dtype=self.dtype)
        
        # Y-direction tridiagonal matrix
        self.Ly_diag = torch.full((self.ny,), -2.0 / self.dy**2, 
                                   device=self.device, dtype=self.dtype)
        self.Ly_off = torch.full((self.ny - 1,), 1.0 / self.dy**2, 
                                  device=self.device, dtype=self.dtype)
        
        # Apply boundary conditions
        if self.bc == "dirichlet":
            self.Lx_diag[0] = 0.0
            self.Lx_diag[-1] = 0.0
            self.Lx_off[0] = 0.0
            self.Lx_off[-1] = 0.0
            
            self.Ly_diag[0] = 0.0
            self.Ly_diag[-1] = 0.0
            self.Ly_off[0] = 0.0
            self.Ly_off[-1] = 0.0
        elif self.bc == "neumann":
            self.Lx_diag[0] = -2.0 / self.dx**2
            self.Lx_diag[-1] = -2.0 / self.dx**2
            self.Lx_off[0] = 2.0 / self.dx**2
            self.Lx_off[-1] = 2.0 / self.dx**2
            
            self.Ly_diag[0] = -2.0 / self.dy**2
            self.Ly_diag[-1] = -2.0 / self.dy**2
            self.Ly_off[0] = 2.0 / self.dy**2
            self.Ly_off[-1] = 2.0 / self.dy**2
    
    def _tridiag_solve(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, 
                       d: torch.Tensor) -> torch.Tensor:
        """
        Solve tridiagonal system Ax = d using Thomas algorithm (GPU-optimized).
        
        Args:
            a: Lower diagonal (size n-1)
            b: Main diagonal (size n)
            c: Upper diagonal (size n-1)
            d: RHS (size [..., n])
        
        Returns:
            Solution x (same shape as d)
        """
        n = b.shape[0]
        batch_shape = d.shape[:-1]
        
        # Forward elimination
        c_prime = torch.zeros_like(b)
        d_prime = torch.zeros_like(d)
        
        c_prime[0] = c[0] / b[0]
        d_prime[..., 0] = d[..., 0] / b[0]
        
        for i in range(1, n - 1):
            denom = b[i] - a[i-1] * c_prime[i-1]
            c_prime[i] = c[i] / denom
            d_prime[..., i] = (d[..., i] - a[i-1] * d_prime[..., i-1]) / denom
        
        i = n - 1
        denom = b[i] - a[i-1] * c_prime[i-1]
        d_prime[..., i] = (d[..., i] - a[i-1] * d_prime[..., i-1]) / denom
        
        # Back substitution
        x = torch.zeros_like(d)
        x[..., -1] = d_prime[..., -1]
        for i in range(n - 2, -1, -1):
            x[..., i] = d_prime[..., i] - c_prime[i] * x[..., i + 1]
        
        return x
    
    def _source_field(self, t: float, sources: List[Dict]) -> torch.Tensor:
        """
        Assemble source field using Gaussian spreading.
        
        Args:
            t: Current time
            sources: List of source dictionaries
        
        Returns:
            Source field S of shape (nx, ny)
        """
        S = torch.zeros((self.nx, self.ny), device=self.device, dtype=self.dtype)
        
        if not sources:
            return S
        
        for src in sources:
            t0, t1 = src["on"]
            if not (t0 <= t <= t1):
                continue
            
            Q = src.get("Q", src.get("q", 0.0))
            if callable(Q):
                Q = Q(t)
            if Q == 0.0:
                continue
            
            x0 = float(src["x"])
            y0 = float(src["y"])
            sigma = src.get("sigma")
            
            if sigma is None:
                continue
            
            # Gaussian distribution
            r_sq = (self.X - x0) ** 2 + (self.Y - y0) ** 2
            G = torch.exp(-r_sq / (2.0 * sigma**2))
            integral_G = torch.sum(G) * self.dx * self.dy
            
            if integral_G > 0:
                S += G * Q / integral_G
        
        return S
    
    def solve(
        self,
        dt: float,
        nt: int,
        T0: Optional[float] = None,
        sources: Optional[List[Dict]] = None,
        store_every: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve heat equation using ADI method on GPU.
        
        Args:
            dt: Time step
            nt: Number of time steps
            T0: Initial temperature
            sources: List of heat sources
            store_every: Storage frequency
        
        Returns:
            times: Array of time points
            Us: Array of temperature fields
        """
        if sources is None:
            sources = []
        
        # Preprocess sources
        t_end = nt * dt
        max_ds = max(self.dx, self.dy)
        auto_sigma = 2.5 * max_ds
        
        processed_sources = []
        for s in sources:
            s_copy = s.copy()
            if "on" not in s_copy or s_copy["on"] is None:
                s_copy["on"] = (0.0, t_end)
            if "sigma" not in s_copy or s_copy["sigma"] is None:
                s_copy["sigma"] = auto_sigma
            processed_sources.append(s_copy)
        sources = processed_sources
        
        # Initialize temperature field
        if T0 is None:
            U = torch.zeros((self.nx, self.ny), device=self.device, dtype=self.dtype)
        else:
            U = torch.full((self.nx, self.ny), T0, device=self.device, dtype=self.dtype)
        
        r = self.kappa * dt / 2.0
        
        # ADI matrices: (I - r*L)
        Ax_diag = 1.0 - r * self.Lx_diag
        Ax_off = -r * self.Lx_off
        
        Ay_diag = 1.0 - r * self.Ly_diag
        Ay_off = -r * self.Ly_off
        
        times, Us = [], []
        
        def add_store(n, t, U):
            if n % store_every == 0:
                times.append(t)
                Us.append(U.cpu().numpy().copy())
        
        t = 0.0
        add_store(0, t, U)
        
        # Main ADI time-stepping loop
        for n in range(nt):
            t_half = t + dt / 2.0
            S_field = self._source_field(t_half, sources) * dt / 2.0
            
            # Step 1: Implicit in X, Explicit in Y
            RHS_x = U.clone()
            # Apply Ly operator explicitly
            for i in range(self.nx):
                laplacian_y = (
                    self.Ly_diag * U[i, :] +
                    torch.cat([self.Ly_off * U[i, 1:], torch.zeros(1, device=self.device, dtype=self.dtype)]) +
                    torch.cat([torch.zeros(1, device=self.device, dtype=self.dtype), self.Ly_off * U[i, :-1]])
                )
                RHS_x[i, :] += r * laplacian_y
            RHS_x += S_field
            
            # Solve tridiagonal system for each y-index
            U_star = torch.zeros_like(U)
            for j in range(self.ny):
                U_star[:, j] = self._tridiag_solve(
                    Ax_off, Ax_diag, Ax_off, RHS_x[:, j]
                )
            
            # Step 2: Implicit in Y, Explicit in X
            RHS_y = U_star.clone()
            # Apply Lx operator explicitly
            for j in range(self.ny):
                laplacian_x = (
                    self.Lx_diag * U_star[:, j] +
                    torch.cat([self.Lx_off * U_star[1:, j], torch.zeros(1, device=self.device, dtype=self.dtype)]) +
                    torch.cat([torch.zeros(1, device=self.device, dtype=self.dtype), self.Lx_off * U_star[:-1, j]])
                )
                RHS_y[:, j] += r * laplacian_x
            RHS_y += S_field
            
            # Solve tridiagonal system for each x-index
            U_next = torch.zeros_like(U)
            for i in range(self.nx):
                U_next[i, :] = self._tridiag_solve(
                    Ay_off, Ay_diag, Ay_off, RHS_y[i, :]
                )
            
            U = U_next
            t += dt
            add_store(n + 1, t, U)
        
        return np.array(times), np.array(Us)
    
    def sample_sensors(
        self, 
        U_field: Union[torch.Tensor, np.ndarray], 
        sensor_xy: Union[torch.Tensor, np.ndarray]
    ) -> np.ndarray:
        """
        Bilinear interpolation of temperature at sensor locations.
        
        Args:
            U_field: Temperature field (nx, ny) - tensor or numpy
            sensor_xy: Sensor coordinates (m, 2) - tensor or numpy
        
        Returns:
            Temperature at sensor locations (m,) as numpy array
        """
        # Convert to tensors if needed
        if isinstance(U_field, np.ndarray):
            U_field = torch.from_numpy(U_field).to(self.device).to(self.dtype)
        if isinstance(sensor_xy, np.ndarray):
            sensor_xy = torch.from_numpy(sensor_xy).to(self.device).to(self.dtype)
        
        m = sensor_xy.shape[0]
        out = torch.zeros(m, device=self.device, dtype=self.dtype)
        
        for i in range(m):
            sx, sy = sensor_xy[i]
            fx = sx / self.dx
            fy = sy / self.dy
            
            ix = torch.clamp(torch.floor(fx).long(), 0, self.nx - 2)
            iy = torch.clamp(torch.floor(fy).long(), 0, self.ny - 2)
            
            tx = fx - ix
            ty = fy - iy
            
            # Bilinear interpolation
            v = (
                (1 - tx) * (1 - ty) * U_field[ix, iy] +
                tx * (1 - ty) * U_field[ix + 1, iy] +
                (1 - tx) * ty * U_field[ix, iy + 1] +
                tx * ty * U_field[ix + 1, iy + 1]
            )
            out[i] = v
        
        return out.cpu().numpy()
    
    def solve_batch(
        self,
        dt: float,
        nt: int,
        sources_batch: List[List[Dict]],
        T0: Optional[float] = None,
        store_every: int = 1,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Solve multiple simulations in parallel (useful for RL batch training).
        
        Args:
            dt: Time step
            nt: Number of time steps
            sources_batch: List of source lists (one per simulation)
            T0: Initial temperature
            store_every: Storage frequency
        
        Returns:
            List of (times, Us) tuples for each simulation
        """
        results = []
        for sources in sources_batch:
            times, Us = self.solve(dt, nt, T0, sources, store_every)
            results.append((times, Us))
        return results


# Utility function for compatibility with original API
def create_solver(Lx, Ly, nx, ny, kappa, bc="dirichlet", use_gpu=True):
    """
    Factory function to create appropriate solver.
    
    Args:
        use_gpu: If True, returns GPU solver; otherwise returns original scipy solver
    """
    if use_gpu:
        return Heat2DGPU(Lx, Ly, nx, ny, kappa, bc=bc)
    else:
        from simulator import Heat2D
        return Heat2D(Lx, Ly, nx, ny, kappa, bc=bc)