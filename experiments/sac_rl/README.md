# Experiment: Soft Actor-Critic (SAC) for source search

This was an early exploration that framed source inversion as a reinforcement-learning problem. It was **not** used in the final submission. The Transformer + BO pipeline in the repo root replaced it.

## Idea

- **Environment** (`HeatInverseEnvGPU` in `SAC_RL.ipynb`): each episode draws a sample and starts from random source parameters θ = (x, y, q) for each source.
- **Action**: a bounded update Δθ.
- **Reward**: the decrease in the forward RMSE between simulated and observed sensor readings, scaled ×100.
- **Observation**: residual statistics, per-sensor mean residuals, the current θ, a source mask, and the current error.
- **Agent**: SAC from stable-baselines3. It was trained on 1,000 simulated samples.
- **Solver**: to make the thousands of simulator calls an RL agent needs affordable, `simulator_gpu.py` ports the ADI heat solver to PyTorch with batching and GPU support. `heat2d_gpu.py` is a simpler explicit-scheme prototype.

## Why it was dropped

Every environment step needs a full forward simulation, so sample efficiency was poor. Per-sample local optimizers (BO and L-BFGS-B), started from a learned initial guess, converged far faster under the challenge's 1-hour inference budget.

## Running

Run from this folder. The notebooks add the repo root to `sys.path` so they can import `utils.py`.

```bash
pip install -r ../../requirements.txt -r requirements.txt
```

1. `dataset_generation.ipynb` writes `data/training_dataset.pkl`.
2. `SAC_RL.ipynb` trains the agent. TensorBoard logs go to `tensorboard_logs/`.
