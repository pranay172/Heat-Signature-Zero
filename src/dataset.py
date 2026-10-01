"""
Dataset and data generation utilities for Heat Signature Zero.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
import pickle
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import make_training_dataset


# ==============================================================================
# Constants (from starter notebook)
# ==============================================================================
LX, LY = 2.0, 1.0
NX, NY = 100, 50
DT = 0.004
Q_RANGE = (0.5, 2.0)
K_MAX = 3  # Maximum number of sources


# ==============================================================================
# Data Generation
# ==============================================================================
def generate_dataset(
    num_samples: int = 20000,
    output_path: Optional[str] = None,
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Generate synthetic dataset using the provided simulator.
    
    Args:
        num_samples: Number of samples to generate
        output_path: Path to save the dataset (pickle format)
        seed: Random seed for reproducibility
        verbose: Print progress
    
    Returns:
        Dataset dictionary with 'samples' and 'meta' keys
    """
    if verbose:
        print(f"Generating {num_samples} samples...")
    
    # Parameters from starter notebook
    num_sources_sensors = {(1, 3): 0.2, (1, 4): 0.2, (2, 5): 0.6}
    noise_std = {0.0: 0.5, 0.1: 0.5}
    nt = {800: 0.6, 1000: 0.2, 1200: 0.2}
    kappa = {0.05: 0.5, 0.1: 0.5}
    bc = "random"
    T0 = None
    
    dataset = make_training_dataset(
        Lx=LX,
        Ly=LY,
        nx=NX,
        ny=NY,
        num_samples=num_samples,
        dt=DT,
        num_sources_sensors=num_sources_sensors,
        q_range=Q_RANGE,
        noise_std=noise_std,
        nt=nt,
        kappa=kappa,
        bc=bc,
        T0=T0,
        seed=seed,
    )
    
    if output_path:
        with open(output_path, 'wb') as f:
            pickle.dump(dataset, f)
        if verbose:
            print(f"Dataset saved to {output_path}")
    
    return dataset


def load_dataset(path: str) -> Dict:
    """Load dataset from pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


# ==============================================================================
# Preprocessing / Normalization
# ==============================================================================
class Normalizer:
    """
    Handles normalization and denormalization of data.
    """
    def __init__(self):
        self.Lx = LX
        self.Ly = LY
        self.q_min, self.q_max = Q_RANGE
    
    def normalize_sensor_positions(self, sensor_xy: np.ndarray) -> np.ndarray:
        """Normalize sensor (x, y) to [0, 1] range."""
        normalized = sensor_xy.copy()
        normalized[:, 0] = normalized[:, 0] / self.Lx
        normalized[:, 1] = normalized[:, 1] / self.Ly
        return normalized
    
    def normalize_sources(self, sources: np.ndarray) -> np.ndarray:
        """
        Normalize source parameters.
        sources: (K, 3) array with columns [x, y, q]
        Returns: (K, 3) array with all values in [0, 1]
        """
        normalized = sources.copy()
        normalized[:, 0] = normalized[:, 0] / self.Lx
        normalized[:, 1] = normalized[:, 1] / self.Ly
        normalized[:, 2] = (normalized[:, 2] - self.q_min) / (self.q_max - self.q_min)
        return normalized
    
    def denormalize_sources(self, sources: np.ndarray) -> np.ndarray:
        """
        Denormalize source parameters back to original scale.
        sources: (K, 3) array with all values in [0, 1]
        Returns: (K, 3) array with original scale
        """
        denormalized = sources.copy()
        denormalized[:, 0] = denormalized[:, 0] * self.Lx
        denormalized[:, 1] = denormalized[:, 1] * self.Ly
        denormalized[:, 2] = denormalized[:, 2] * (self.q_max - self.q_min) + self.q_min
        return denormalized
    
    def normalize_time_series(self, Y: np.ndarray, per_sample: bool = True) -> Tuple[np.ndarray, float, float]:
        """
        Normalize sensor readings.
        Y: (T, M) sensor readings
        Returns: normalized Y, mean, std
        """
        if per_sample:
            mean = Y.mean()
            std = Y.std() + 1e-8  # Avoid division by zero
        else:
            mean, std = 0.0, 1.0
        return (Y - mean) / std, mean, std


# ==============================================================================
# PyTorch Dataset
# ==============================================================================
class HeatSourceDataset(Dataset):
    """
    PyTorch Dataset for heat source estimation.
    
    Each sample contains:
        - Y_noisy: (T, M) sensor readings (normalized)
        - sensor_xy: (M, 2) sensor positions (normalized to [0,1])
        - metadata: dict with kappa, bc, T0, nt
        - n_sources: number of sources
        - sources: (K, 3) ground truth source parameters (normalized)
    """
    
    def __init__(
        self,
        dataset: Dict,
        normalize: bool = True,
        max_time_steps: Optional[int] = None,
    ):
        """
        Args:
            dataset: Dataset dict from make_training_dataset or load_dataset
            normalize: Whether to normalize inputs and targets
            max_time_steps: Truncate/pad time series to this length (None = no change)
        """
        self.samples = dataset['samples']
        self.meta = dataset.get('meta', {'dt': DT, 'q_range': Q_RANGE})
        self.normalize = normalize
        self.max_time_steps = max_time_steps
        self.normalizer = Normalizer()
        
        # Precompute max sensors and sources for this dataset
        self.max_sensors = max(s['sensors_xy'].shape[0] for s in self.samples)
        self.max_sources = max(s['n_sources'] for s in self.samples)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Extract data
        Y_noisy = sample['Y_noisy'].astype(np.float32)  # (T, M)
        sensor_xy = sample['sensors_xy'].astype(np.float32)  # (M, 2)
        n_sources = sample['n_sources']
        
        # Extract ground truth sources
        true_sources = sample.get('true_sources', None)
        if true_sources is not None:
            sources = np.array([
                [s['x'], s['y'], s['q']] for s in true_sources
            ], dtype=np.float32)  # (K, 3)
        else:
            sources = None
        
        # Extract metadata
        sample_meta = sample.get('sample_metadata', {})
        kappa = float(sample_meta.get('kappa', 0.1))
        bc = sample_meta.get('bc', 'dirichlet')
        T0 = float(sample_meta.get('T0', 0.0))
        nt = int(sample_meta.get('nt', Y_noisy.shape[0]))
        
        
        # Compute global features (before normalization)
        # Features: [max_temp, mean_temp, std_temp, rise_rate]
        # These help the model recover absolute magnitude info lost during normalization
        max_temp = np.max(Y_noisy)
        mean_temp = np.mean(Y_noisy)
        std_temp = np.std(Y_noisy)
        
        # Rise rate: slope of first 10% of time steps
        early_t = max(1, Y_noisy.shape[0] // 10)
        # Average over all sensors
        vals_0 = np.mean(Y_noisy[0, :])
        vals_early = np.mean(Y_noisy[early_t, :])
        rise_rate = (vals_early - vals_0) / (early_t * self.meta['dt'])
        
        global_features = np.array([max_temp, mean_temp, std_temp, rise_rate], dtype=np.float32)
        
        # Normalize if requested
        if self.normalize:
            Y_noisy, _, _ = self.normalizer.normalize_time_series(Y_noisy)
            sensor_xy = self.normalizer.normalize_sensor_positions(sensor_xy)
            if sources is not None:
                sources = self.normalizer.normalize_sources(sources)
        
        # Handle time series length
        T = Y_noisy.shape[0]
        if self.max_time_steps is not None:
            if T > self.max_time_steps:
                # Use log-spaced indices to preserve more samples at the beginning
                # where transient rise provides important source proximity information
                log_indices = np.logspace(0, np.log10(T), self.max_time_steps, dtype=float)
                indices = np.clip(log_indices.astype(int) - 1, 0, T - 1)
                # Ensure unique indices by removing duplicates and filling with linspace
                indices = np.unique(indices)
                if len(indices) < self.max_time_steps:
                    # Fill remaining with linspace
                    remaining = self.max_time_steps - len(indices)
                    extra = np.linspace(0, T - 1, remaining + 2, dtype=int)[1:-1]
                    indices = np.unique(np.concatenate([indices, extra]))[:self.max_time_steps]
                indices = np.sort(indices)
                Y_noisy = Y_noisy[indices]
            elif T < self.max_time_steps:
                # Pad with zeros
                pad = np.zeros((self.max_time_steps - T, Y_noisy.shape[1]), dtype=np.float32)
                Y_noisy = np.concatenate([Y_noisy, pad], axis=0)
        
        # Build output dict
        output = {
            'sample_id': sample['sample_id'],
            'Y_noisy': torch.from_numpy(Y_noisy),  # (T, M)
            'sensor_xy': torch.from_numpy(sensor_xy),  # (M, 2)
            'n_sensors': sensor_xy.shape[0],
            'n_sources': n_sources,
            'kappa': kappa,
            'bc': 0 if bc == 'dirichlet' else 1,  # Encode as int
            'T0': T0,
            'nt': nt,
            'global_features': torch.from_numpy(global_features), # (4,)
        }
        
        if sources is not None:
            output['sources'] = torch.from_numpy(sources)  # (K, 3)
        
        return output


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Custom collate function to handle variable number of sensors and sources.
    
    Pads sensors to max_M in batch and sources to max_K.
    Returns masks to indicate valid sensors/sources.
    """
    batch_size = len(batch)
    
    # Find max dimensions in this batch
    max_T = max(b['Y_noisy'].shape[0] for b in batch)
    max_M = max(b['n_sensors'] for b in batch)
    max_K = max(b['n_sources'] for b in batch)
    
    # Initialize tensors
    Y_noisy = torch.zeros(batch_size, max_T, max_M)
    sensor_xy = torch.zeros(batch_size, max_M, 2)
    sensor_mask = torch.zeros(batch_size, max_M, dtype=torch.bool)
    
    has_sources = 'sources' in batch[0]
    if has_sources:
        sources = torch.zeros(batch_size, max_K, 3)
        source_mask = torch.zeros(batch_size, max_K, dtype=torch.bool)
    
    n_sources_list = []
    kappa_list = []
    bc_list = []
    sample_ids = []
    global_features_list = []
    
    for i, b in enumerate(batch):
        T, M = b['Y_noisy'].shape
        K = b['n_sources']
        
        # Fill padded tensors
        Y_noisy[i, :T, :M] = b['Y_noisy']
        sensor_xy[i, :M] = b['sensor_xy']
        sensor_mask[i, :M] = True
        
        if has_sources:
            sources[i, :K] = b['sources']
            source_mask[i, :K] = True
        
        n_sources_list.append(K)
        kappa_list.append(b['kappa'])
        bc_list.append(b['bc'])
        sample_ids.append(b['sample_id'])
        global_features_list.append(b['global_features'])
    
    output = {
        'sample_ids': sample_ids,
        'Y_noisy': Y_noisy,
        'sensor_xy': sensor_xy,
        'sensor_mask': sensor_mask,
        'n_sources': torch.tensor(n_sources_list),
        'kappa': torch.tensor(kappa_list, dtype=torch.float32),
        'bc': torch.tensor(bc_list, dtype=torch.long),
        'global_features': torch.stack(global_features_list),
    }
    
    if has_sources:
        output['sources'] = sources
        output['source_mask'] = source_mask
    
    return output


def create_dataloaders(
    dataset: Dict,
    batch_size: int = 64,
    val_split: float = 0.1,
    num_workers: int = 4,
    max_time_steps: Optional[int] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        dataset: Dataset dict
        batch_size: Batch size
        val_split: Fraction for validation
        num_workers: Number of data loading workers
        max_time_steps: Max time steps (None = variable)
        seed: Random seed for split
    
    Returns:
        train_loader, val_loader
    """
    # Create full dataset
    full_dataset = HeatSourceDataset(
        dataset, normalize=True, max_time_steps=max_time_steps
    )
    
    # Split into train/val
    n_total = len(full_dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val], generator=generator
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    return train_loader, val_loader


# ==============================================================================
# CLI for data generation
# ==============================================================================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate training dataset')
    parser.add_argument('--num_samples', type=int, default=20000,
                        help='Number of samples to generate')
    parser.add_argument('--output', type=str, default='data/train_dataset.pkl',
                        help='Output path for dataset')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Generate dataset
    generate_dataset(
        num_samples=args.num_samples,
        output_path=args.output,
        seed=args.seed,
        verbose=True,
    )