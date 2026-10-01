"""
Diagnostic script to analyze score components (accuracy vs diversity).
Identifies what is hindering the LB score from reaching the top.
"""

import numpy as np
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from src.model import SourcePredictorTransformer
from src.dataset import Normalizer, LX, LY, Q_RANGE, DT
from src.bo_refiner import BORefiner, compute_forward_loss
from src.gd_refiner import GDRefiner
from src.inference import InferencePipeline

# Competition parameters
N_MAX = 3
LAMBDA = 0.3
TAU = 0.2
SCALE_FACTORS = (LX, LY, Q_RANGE[1])


def score_breakdown(
    gt_dataset: Dict,
    pred_dataset: List[Dict],
    solver_kwargs: Dict,
    verbose: bool = True,
) -> Dict:
    """
    Compute detailed score breakdown for each sample.
    
    Returns dict with:
    - per_sample: list of {sample_id, n_candidates, n_valid, losses, accuracy_component, diversity_component, total}
    - summary: aggregate statistics
    """
    from utils import filter_distinct_candidates, simulate_candidate, forward_error
    
    meta = gt_dataset['meta']
    gt_samples = {s['sample_id']: s for s in gt_dataset['samples']}
    
    results = []
    
    for pred_sample in pred_dataset:
        sample_id = pred_sample['sample_id']
        sample_gt = gt_samples.get(sample_id)
        
        if sample_gt is None:
            print(f"Warning: {sample_id} not found in GT")
            continue
            
        candidates = pred_sample['estimated_sources']
        n_candidates = len(candidates)
        
        # Filter distinct candidates
        valid_candidates = filter_distinct_candidates(candidates, TAU, N_MAX, SCALE_FACTORS)
        n_valid = len(valid_candidates)
        
        # Compute forward loss for each valid candidate
        Y_true = sample_gt['Y']
        losses = []
        for candidate in valid_candidates:
            Y_pred = simulate_candidate(candidate, sample_gt, meta, solver_kwargs)
            fe = forward_error(Y_true, Y_pred, loss='rmse')
            losses.append(fe)
        
        # Score components
        if n_valid > 0:
            accuracy_component = (1 / n_valid) * sum(1 / (1 + L) for L in losses)
            diversity_component = LAMBDA * (n_valid / N_MAX)
            total_score = accuracy_component + diversity_component
        else:
            accuracy_component = 0.0
            diversity_component = 0.0
            total_score = 0.0
        
        result = {
            'sample_id': sample_id,
            'n_candidates': n_candidates,
            'n_valid': n_valid,
            'losses': losses,
            'avg_loss': np.mean(losses) if losses else float('nan'),
            'min_loss': min(losses) if losses else float('nan'),
            'accuracy_component': accuracy_component,
            'diversity_component': diversity_component,
            'total_score': total_score,
        }
        results.append(result)
        
        if verbose:
            print(f"\n{sample_id}:")
            print(f"  Candidates: {n_candidates} -> Valid: {n_valid}")
            print(f"  Losses: {[f'{l:.4f}' for l in losses]}")
            print(f"  Accuracy: {accuracy_component:.4f}, Diversity: {diversity_component:.4f}")
            print(f"  Total: {total_score:.4f}")
    
    # Summary statistics
    all_n_valid = [r['n_valid'] for r in results]
    all_avg_loss = [r['avg_loss'] for r in results if not np.isnan(r['avg_loss'])]
    all_min_loss = [r['min_loss'] for r in results if not np.isnan(r['min_loss'])]
    all_accuracy = [r['accuracy_component'] for r in results]
    all_diversity = [r['diversity_component'] for r in results]
    all_total = [r['total_score'] for r in results]
    
    summary = {
        'n_samples': len(results),
        'avg_n_valid': np.mean(all_n_valid),
        'samples_with_1_valid': sum(1 for n in all_n_valid if n == 1),
        'samples_with_2_valid': sum(1 for n in all_n_valid if n == 2),
        'samples_with_3_valid': sum(1 for n in all_n_valid if n == 3),
        'avg_avg_loss': np.mean(all_avg_loss),
        'avg_min_loss': np.mean(all_min_loss),
        'avg_accuracy_component': np.mean(all_accuracy),
        'avg_diversity_component': np.mean(all_diversity),
        'avg_total_score': np.mean(all_total),
        'max_accuracy_possible': 1.0,
        'max_diversity_possible': LAMBDA,
    }
    
    return {'per_sample': results, 'summary': summary}


def analyze_diversity_failure(
    gt_dataset: Dict,
    pred_dataset: List[Dict],
    verbose: bool = True,
) -> Dict:
    """
    Analyze WHY candidates are being filtered out.
    
    Checks:
    1. Are candidates too similar (fail tau threshold)?
    2. Do we not generate enough candidates?
    """
    from utils import scale_sources_fixed, sort_sources_candidate, dissimilarity
    
    failure_reasons = {
        'too_few_candidates': 0,
        'too_similar': 0,
        'successful': 0,
    }
    
    similarity_details = []
    
    for pred_sample in pred_dataset:
        sample_id = pred_sample['sample_id']
        candidates = pred_sample['estimated_sources']
        n_candidates = len(candidates)
        
        if n_candidates < N_MAX:
            failure_reasons['too_few_candidates'] += 1
            
        # Check pairwise similarities
        scaled_candidates = scale_sources_fixed(candidates, SCALE_FACTORS)
        normalized_candidates = [sort_sources_candidate(c) for c in scaled_candidates]
        
        # Build similarity matrix
        similarities = []
        for i in range(len(normalized_candidates)):
            for j in range(i+1, len(normalized_candidates)):
                d = dissimilarity(normalized_candidates[i], normalized_candidates[j])
                similarities.append({
                    'sample_id': sample_id,
                    'pair': (i, j),
                    'dissimilarity': d,
                    'is_distinct': d > TAU,
                })
        
        # Count how many pairs are too similar
        too_similar_count = sum(1 for s in similarities if not s['is_distinct'])
        if too_similar_count > 0:
            failure_reasons['too_similar'] += 1
            
        if n_candidates >= N_MAX and too_similar_count == 0:
            failure_reasons['successful'] += 1
            
        similarity_details.extend(similarities)
        
        if verbose and similarities:
            print(f"\n{sample_id}:")
            print(f"  Submitted: {n_candidates} candidates")
            for s in similarities:
                status = "OK" if s['is_distinct'] else "TOO SIMILAR"
                print(f"    Pair {s['pair']}: dissimilarity={s['dissimilarity']:.4f} [{status}]")
    
    return {
        'failure_reasons': failure_reasons,
        'similarity_details': similarity_details,
        'avg_dissimilarity': np.mean([s['dissimilarity'] for s in similarity_details]) if similarity_details else 0,
    }


def run_inference_and_analyze(
    model_path: str,
    test_data_path: str,
    gt_data_path: str,
    method: str = 'gd',
    n_search_candidates: int = 10,
    n_calls: int = 30,
    device: str = 'cuda',
    limit: int = 0,
):
    """Run inference and analyze results."""
    import torch
    
    # Check device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Load data
    print(f"Loading test data from {test_data_path}...")
    with open(test_data_path, 'rb') as f:
        test_dataset = pickle.load(f)
    
    # Handle both list and dict formats
    if isinstance(test_dataset, list):
        test_dataset = {'samples': test_dataset, 'meta': {'dt': DT, 'q_range': Q_RANGE}}
    
    print(f"Loading GT data from {gt_data_path}...")
    with open(gt_data_path, 'rb') as f:
        gt_dataset = pickle.load(f)
    
    # Handle both list and dict formats
    if isinstance(gt_dataset, list):
        gt_dataset = {'samples': gt_dataset, 'meta': {'dt': DT, 'q_range': Q_RANGE}}
        
    # Apply limit
    if limit > 0:
        print(f"Limiting to first {limit} samples")
        test_dataset['samples'] = test_dataset['samples'][:limit]
        
        # Filter GT to match sample_ids
        sample_ids = set(s['sample_id'] for s in test_dataset['samples'])
        gt_dataset['samples'] = [s for s in gt_dataset['samples'] if s['sample_id'] in sample_ids]
    
    print(f"Loaded {len(test_dataset['samples'])} samples")
    
    # Create pipeline
    pipeline = InferencePipeline(
        model_path=model_path,
        device=device,
        n_candidates=3,
        n_search_candidates=n_search_candidates,
        n_calls=n_calls,
        n_calls_gd=20,
        n_calls_bo=10,
        method=method,
    )
    
    # Run inference
    print(f"\nRunning inference with {method} method...")
    submission = pipeline.run_inference(test_dataset, output_path='/tmp/temp_submission.npz', verbose=True)
    
    # Analyze score breakdown
    print("\n" + "="*60)
    print("SCORE BREAKDOWN ANALYSIS")
    print("="*60)
    
    solver_kwargs = {'Lx': LX, 'Ly': LY, 'nx': 100, 'ny': 50}
    breakdown = score_breakdown(gt_dataset, submission['samples'], solver_kwargs, verbose=True)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    s = breakdown['summary']
    print(f"Number of samples: {s['n_samples']}")
    print(f"\nDiversity Analysis:")
    print(f"  Avg N_valid: {s['avg_n_valid']:.2f} / 3.00")
    print(f"  Samples with 1 valid: {s['samples_with_1_valid']} ({100*s['samples_with_1_valid']/s['n_samples']:.1f}%)")
    print(f"  Samples with 2 valid: {s['samples_with_2_valid']} ({100*s['samples_with_2_valid']/s['n_samples']:.1f}%)")
    print(f"  Samples with 3 valid: {s['samples_with_3_valid']} ({100*s['samples_with_3_valid']/s['n_samples']:.1f}%)")
    print(f"\nAccuracy Analysis:")
    print(f"  Avg forward loss (RMSE): {s['avg_avg_loss']:.4f}")
    print(f"  Avg min loss per sample: {s['avg_min_loss']:.4f}")
    print(f"\nScore Components:")
    print(f"  Avg Accuracy Component: {s['avg_accuracy_component']:.4f} (max possible: {s['max_accuracy_possible']:.2f})")
    print(f"  Avg Diversity Component: {s['avg_diversity_component']:.4f} (max possible: {s['max_diversity_possible']:.2f})")
    print(f"  Avg Total Score: {s['avg_total_score']:.4f} (max possible: 1.30)")
    
    # Calculate gaps
    accuracy_gap = s['max_accuracy_possible'] - s['avg_accuracy_component']
    diversity_gap = s['max_diversity_possible'] - s['avg_diversity_component']
    
    print(f"\n" + "="*60)
    print("BOTTLENECK IDENTIFICATION")
    print("="*60)
    print(f"  Accuracy Gap: {accuracy_gap:.4f} ({100*accuracy_gap/1.3:.1f}% of total possible score)")
    print(f"  Diversity Gap: {diversity_gap:.4f} ({100*diversity_gap/1.3:.1f}% of total possible score)")
    
    if accuracy_gap > diversity_gap:
        print(f"\n  >>> BOTTLENECK: ACCURACY <<<")
        print(f"      The forward loss (RMSE) is too high.")
        print(f"      Focus on: Better refinement, more n_calls, or better NN predictions.")
    else:
        print(f"\n  >>> BOTTLENECK: DIVERSITY <<<")
        print(f"      Not enough distinct valid candidates.")
        print(f"      Focus on: Generating more diverse starting points, or adjusting search.")
    
    # Analyze diversity failures
    print("\n" + "="*60)
    print("DIVERSITY FAILURE ANALYSIS")
    print("="*60)
    diversity_analysis = analyze_diversity_failure(gt_dataset, submission['samples'], verbose=False)
    fr = diversity_analysis['failure_reasons']
    print(f"  Samples with <3 submitted candidates: {fr['too_few_candidates']}")
    print(f"  Samples with candidates too similar: {fr['too_similar']}")
    print(f"  Samples with 3 distinct candidates: {fr['successful']}")
    print(f"  Average pairwise dissimilarity: {diversity_analysis['avg_dissimilarity']:.4f} (threshold: {TAU})")
    
    return breakdown, diversity_analysis


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze score components')
    parser.add_argument('--model_path', type=str, default='checkpoints/best.pt')
    parser.add_argument('--test_data', type=str, default='data/validation_dataset_40.pkl')
    parser.add_argument('--gt_data', type=str, default='data/validation_dataset_40.pkl')
    parser.add_argument('--method', type=str, default='hybrid', choices=['gd', 'bo', 'hybrid'])
    parser.add_argument('--n_search_candidates', type=int, default=10)
    parser.add_argument('--n_calls', type=int, default=30)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of samples to process')
    
    args = parser.parse_args()
    
    run_inference_and_analyze(
        model_path=args.model_path,
        test_data_path=args.test_data,
        gt_data_path=args.gt_data,
        method=args.method,
        n_search_candidates=args.n_search_candidates,
        n_calls=args.n_calls,
        device=args.device,
        limit=args.limit,
    )