"""
Complete inference pipeline for Heat Signature Zero.
Combines neural network prediction with refinement (BO or Gradient Descent).
"""

import numpy as np
import torch
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import argparse
import time
import sys

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import SourcePredictorTransformer
from src.dataset import Normalizer, LX, LY, Q_RANGE, DT
from src.bo_refiner import BORefiner, compute_forward_loss
from src.gd_refiner import GDRefiner


class InferencePipeline:
    """
    Complete inference pipeline combining NN + Refinement.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        n_candidates: int = 3,
        n_search_candidates: Optional[int] = None,
        n_calls: int = 20,
        solver_kwargs: Optional[Dict] = None,
        n_jobs: int = -1,
        method: str = 'hybrid', # 'bo', 'gd', or 'hybrid'
        n_calls_gd: int = 15,
        n_calls_bo: int = 15,
    ):
        """
        Args:
            model_path: Path to model checkpoint
            device: Device for inference
            n_candidates: Number of candidates per sample (output)
            n_search_candidates: Number of candidates to generate during SEARCH
            n_calls: calls/steps per candidate (total budget per stage if not specified)
            solver_kwargs: Solver configuration
            n_jobs: Number of parallel jobs for BO (-1 for all cores)
            method: 'bo', 'gd', or 'hybrid'
            n_calls_gd: steps for GD phase (if hybrid)
            n_calls_bo: steps for BO phase (if hybrid)
        """
        self.device = device
        self.n_candidates = n_candidates
        self.n_search_candidates = n_search_candidates or n_candidates
        self.n_calls = n_calls
        self.n_calls_gd = n_calls_gd
        self.n_calls_bo = n_calls_bo
        self.n_jobs = n_jobs
        self.method = method
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = SourcePredictorTransformer.load(model_path, device='cpu')
        self.model = self.model.to(device)
        self.model.eval()
        
        # Normalizer
        self.normalizer = Normalizer()
        
        # Solver / Refiner Setup
        self.solver_kwargs = solver_kwargs or {'Lx': LX, 'Ly': LY, 'nx': 100, 'ny': 50}
        
        # Initialize BO Refiner (used for 'bo' method AND for start points in 'gd')
        self.bo_refiner = BORefiner(
            Lx=LX,
            Ly=LY,
            q_range=Q_RANGE,
            solver_kwargs=self.solver_kwargs,
            dt=DT,
        )
        
        # Initialize GD Refiner if needed
        if method in ['gd', 'hybrid']:
            self.gd_refiner = GDRefiner(
                Lx=LX, Ly=LY, 
                q_range=Q_RANGE, 
                nx=100, ny=50, 
                dt=DT,
                device=device
            )
    
    def preprocess_sample(self, sample: Dict) -> Tuple[torch.Tensor, ...]:
        """Preprocess a single sample for model inference."""
        Y_noisy = sample['Y_noisy'].astype(np.float32)  # (T, M)
        sensor_xy = sample['sensors_xy'].astype(np.float32)  # (M, 2)
        n_sources = sample['n_sources']
        
        meta = sample.get('sample_metadata', {})
        kappa = float(meta.get('kappa', 0.1))
        bc = meta.get('bc', 'dirichlet')
        
        # Compute global features
        max_temp = np.max(Y_noisy)
        mean_temp = np.mean(Y_noisy)
        std_temp = np.std(Y_noisy)
        
        # Rise rate: slope of first 10% of time steps
        early_t = max(1, Y_noisy.shape[0] // 10)
        vals_0 = np.mean(Y_noisy[0, :])
        vals_early = np.mean(Y_noisy[early_t, :])
        rise_rate = (vals_early - vals_0) / (early_t * DT)
        
        global_features = np.array([max_temp, mean_temp, std_temp, rise_rate], dtype=np.float32)
        
        # Normalize
        Y_noisy_norm, _, _ = self.normalizer.normalize_time_series(Y_noisy)
        sensor_xy_norm = self.normalizer.normalize_sensor_positions(sensor_xy)
        
        # Convert to tensors
        Y_noisy_t = torch.from_numpy(Y_noisy_norm).unsqueeze(0)  # (1, T, M)
        sensor_xy_t = torch.from_numpy(sensor_xy_norm).unsqueeze(0)  # (1, M, 2)
        sensor_mask = torch.ones(1, sensor_xy.shape[0], dtype=torch.bool)
        kappa_t = torch.tensor([kappa], dtype=torch.float32)
        bc_t = torch.tensor([0 if bc == 'dirichlet' else 1], dtype=torch.long)
        n_sources_t = torch.tensor([n_sources])
        global_features_t = torch.from_numpy(global_features).unsqueeze(0) # (1, 4)
        
        return Y_noisy_t, sensor_xy_t, sensor_mask, kappa_t, bc_t, n_sources_t, global_features_t
    
    @torch.no_grad()
    def predict_nn(self, sample: Dict) -> np.ndarray:
        """Get NN prediction for a sample."""
        Y_noisy, sensor_xy, sensor_mask, kappa, bc, n_sources, global_features = self.preprocess_sample(sample)
        
        Y_noisy = Y_noisy.to(self.device)
        sensor_xy = sensor_xy.to(self.device)
        sensor_mask = sensor_mask.to(self.device)
        kappa = kappa.to(self.device)
        bc = bc.to(self.device)
        global_features = global_features.to(self.device)
        
        # Forward pass
        pred = self.model(Y_noisy, sensor_xy, sensor_mask, kappa, bc, n_sources, global_features)  # (1, K_max, 3)
        
        # Get relevant sources and denormalize
        K = sample['n_sources']
        pred = pred[0, :K].cpu().numpy()
        pred = self.normalizer.denormalize_sources(pred)
        
        return pred
    
    def predict_sample(
        self,
        sample: Dict,
        seed: int = 42,
    ) -> List[List[Tuple[float, float, float]]]:
        """Full inference for one sample."""
        nn_pred = self.predict_nn(sample)
        
        # Refinement logic
        if self.method == 'hybrid':
             # --------------------------
             # Hybrid Approach (GD -> BO)
             # --------------------------
             rng = np.random.default_rng(seed)
             
             # Step 1: Generate Physics-Informed Starts
             candidates_start_arrays = self.bo_refiner.generate_physics_informed_starts(
                nn_pred,
                sample,
                n_starts=self.n_search_candidates,
                rng=rng
             )
             
             starts_list = []
             for s in candidates_start_arrays:
                 cand = []
                 for k in range(s.shape[0]):
                     cand.append((float(s[k,0]), float(s[k,1]), float(s[k,2])))
                 starts_list.append(cand)
                 
             # Step 2: GD Refinement (Basin Finding)
             # Moves candidates to valid regions
             gd_refined_list = self.gd_refiner.refine(
                 starts_list,
                 sample,
                 n_steps=self.n_calls_gd,
                 verbose=False,
                 patience=5
             )
             
             # Step 2.5: Evaluate and Select Top Candidates
             # Compute loss for all GD candidates to pick the best basins
             gd_candidates_with_loss = []
             for cand_list in gd_refined_list:
                 try:
                    loss = compute_forward_loss(cand_list, sample, self.solver_kwargs, DT)
                 except Exception:
                    loss = 1e9
                 gd_candidates_with_loss.append((cand_list, loss))
             
             # Sort by loss
             gd_candidates_with_loss.sort(key=lambda x: x[1])
             
             # Prepare for filtering (needs list of (ndarray, loss))
             gd_candidates_for_filter = [
                 (np.array(c[0]), c[1]) for c in gd_candidates_with_loss
             ]
             
             # Filter distinct candidates
             # tau=0.2 is reasonable for this domain (as per user analysis)
             distinct_candidates_arr = self.bo_refiner.filter_distinct(
                 gd_candidates_for_filter, 
                 tau=0.2, 
                 n_max=min(5, self.n_search_candidates)
             )
             
             # Use distinct candidates for refinement
             top_gd_candidates = distinct_candidates_arr
             
             # Step 3: BO Refinement (Polishing)
             # Fine-tune each GD result to maximize accuracy
             from joblib import Parallel, delayed
             
             def refine_one_candidate(sources, seed_offset):
                 # Convert list of tuples back to array for BO
                 arr = np.array(sources)
                 
                 # Clip to bounds to prevent scikit-optimize/scipy errors
                 # Margin must match what BORefiner uses internally
                 margin = self.bo_refiner.margin
                 arr[:, 0] = np.clip(arr[:, 0], margin * self.bo_refiner.Lx, (1 - margin) * self.bo_refiner.Lx)
                 arr[:, 1] = np.clip(arr[:, 1], margin * self.bo_refiner.Ly, (1 - margin) * self.bo_refiner.Ly)
                 arr[:, 2] = np.clip(arr[:, 2], self.bo_refiner.q_min, self.bo_refiner.q_max)
                 
                 # Run BO on this single candidate basin
                 refined_arr, loss = self.bo_refiner.refine_hybrid(
                     arr, 
                     sample, 
                     n_initial=0,
                     n_refine=self.n_calls_bo, 
                     random_state=seed + seed_offset
                 )
                 return refined_arr, loss

             # Run in parallel if n_jobs != 1
             # Note: GD is already fast/gpu, BO is cpu bound
             refined_results = Parallel(n_jobs=self.n_jobs)(
                 delayed(refine_one_candidate)(cand, i) 
                 for i, cand in enumerate(top_gd_candidates)
             )
             
             candidates = refined_results
             candidates.sort(key=lambda x: x[1])

        elif self.method == 'gd':
             # --------------------------
             # Gradient Descent Approach
             # --------------------------
             rng = np.random.default_rng(seed)
             
             # Generate start points (Physics-Informed + NN)
             candidates_start_arrays = self.bo_refiner.generate_physics_informed_starts(
                nn_pred,
                sample,
                n_starts=self.n_search_candidates,
                rng=rng
             )
             
             # Convert to list format for GDRefiner
             starts_list = []
             for s in candidates_start_arrays:
                 cand = []
                 for k in range(s.shape[0]):
                     cand.append((float(s[k,0]), float(s[k,1]), float(s[k,2])))
                 starts_list.append(cand)
             
             # Refine with optimized GD (cosine LR, early stopping, etc.)
             refined_list = self.gd_refiner.refine(
                 starts_list,
                 sample,
                 n_steps=self.n_calls,
                 verbose=False
             )
             
             # Evaluate candidates to sort them
             candidates = []
             for cand_list in refined_list:
                 arr = np.array(cand_list)
                 try:
                    loss = compute_forward_loss(cand_list, sample, self.solver_kwargs, DT)
                 except Exception:
                    loss = 1e9
                 candidates.append((arr, loss))
             
             candidates.sort(key=lambda x: x[1])

        else:
            # --------------------------
            # Standard (Adaptive) BO Approach
            # --------------------------
            total_budget = self.n_search_candidates * self.n_calls
            candidates = self.bo_refiner.generate_candidates_adaptive(
                nn_pred,
                sample,
                n_candidates=self.n_candidates,
                total_budget=total_budget,
                coarse_calls=2, # Initial random search breadth
                seed=seed,
                n_jobs=self.n_jobs,
            )
        
        # Filter distinct candidates
        distinct = self.bo_refiner.filter_distinct(candidates, tau=0.2, n_max=self.n_candidates)
        
        # Format result
        result = []
        for sources in distinct:
            candidate = [(float(sources[i, 0]), float(sources[i, 1]), float(sources[i, 2]))
                        for i in range(sources.shape[0])]
            result.append(candidate)
        
        return result
    
    def run_inference(
        self,
        test_dataset: Dict,
        output_path: str = 'submission.npz',
        verbose: bool = True,
    ) -> Dict:
        """Run inference on full test dataset."""
        samples = test_dataset['samples']
        results = []
        
        total_time = 0
        iterator = tqdm(samples, desc='Inference') if verbose else samples
        
        for i, sample in enumerate(iterator):
            start = time.time()
            
            # Run inference
            candidates = self.predict_sample(sample, seed=42 + i)
            
            results.append({
                'sample_id': sample['sample_id'],
                'estimated_sources': candidates,
            })
            
            elapsed = time.time() - start
            total_time += elapsed
            
            if verbose and (i + 1) % 10 == 0:
                avg_time = total_time / (i + 1)
                print(f"\n  Avg time per sample: {avg_time:.2f}s")
        
        # Create submission
        submission = {'samples': results}
        np.savez(output_path, samples=results)
        
        if verbose:
            print(f"\nSubmission saved to {output_path}")
            print(f"Total samples: {len(results)}")
            print(f"Total time: {total_time:.1f}s")
            print(f"Avg time per sample: {total_time / len(results):.2f}s")
        
        return submission


def evaluate_submission(
    submission: Dict,
    gt_dataset: Dict,
    solver_kwargs: Optional[Dict] = None,
    verbose: bool = True,
) -> float:
    """Evaluate submission using the competition scoring function."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils import score_submission
    
    solver_kwargs = solver_kwargs or {'Lx': LX, 'Ly': LY, 'nx': 100, 'ny': 50}
    
    # Score parameters from competition
    N_max = 3
    lambda_ = 0.3
    tau = 0.2
    scale_factors = (LX, LY, Q_RANGE[1])
    forward_loss = 'rmse'
    
    score = score_submission(
        gt_dataset,
        submission['samples'],
        N_max,
        lambda_,
        tau,
        scale_factors,
        forward_loss,
        solver_kwargs,
    )
    
    if verbose:
        print(f"\nFinal Score: {score:.6f}")
    
    return score


def main():
    parser = argparse.ArgumentParser(description='Run inference pipeline')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to test dataset (pickle)')
    parser.add_argument('--output', type=str, default='submission.npz',
                        help='Output submission file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--method', type=str, default='hybrid', choices=['bo', 'gd', 'hybrid'],
                        help="Optimization method: 'bo', 'gd', or 'hybrid'")
    
    parser.add_argument('--n_candidates', type=int, default=3,
                        help='Number of candidates per sample (output)')
    parser.add_argument('--n_search_candidates', type=int, default=10,
                        help='Number of candidates to generate during SEARCH')
    parser.add_argument('--n_calls', type=int, default=30,
                        help='Number of optimization steps/calls per candidate (total budget)')
    parser.add_argument('--n_calls_gd', type=int, default=20,
                        help='Number of GD steps (hybrid only)')
    parser.add_argument('--n_calls_bo', type=int, default=10,
                        help='Number of BO calls (hybrid only)')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of parallel jobs (-1 for all cores)')
    
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate using GT dataset (for testing)')
    parser.add_argument('--gt_data', type=str, default=None,
                        help='Path to GT dataset for evaluation')
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load test dataset
    print(f"Loading test dataset from {args.test_data}...")
    with open(args.test_data, 'rb') as f:
        test_dataset = pickle.load(f)
    print(f"Loaded {len(test_dataset['samples'])} samples")
    
    # Create pipeline
    pipeline = InferencePipeline(
        model_path=args.model_path,
        device=args.device,
        n_candidates=args.n_candidates,
        n_search_candidates=args.n_search_candidates,
        n_calls=args.n_calls,
        n_calls_gd=args.n_calls_gd,
        n_calls_bo=args.n_calls_bo,
        n_jobs=args.n_jobs,
        method=args.method,
    )
    
    # Run inference
    submission = pipeline.run_inference(
        test_dataset,
        output_path=args.output,
        verbose=True,
    )
    
    # Evaluate if GT provided
    if args.evaluate and args.gt_data:
        print("\nEvaluating submission...")
        with open(args.gt_data, 'rb') as f:
            gt_dataset = pickle.load(f)
        score = evaluate_submission(submission, gt_dataset)


if __name__ == '__main__':
    main()