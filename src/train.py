"""
Training pipeline for SourcePredictorTransformer.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import time
from typing import Optional, Dict, Tuple
import argparse
import sys
# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import SourcePredictorTransformer, model_summary
from src.dataset import (
    generate_dataset, load_dataset, create_dataloaders, 
    HeatSourceDataset, collate_fn
)
from src.loss import combined_loss, compute_metrics


class Trainer:
    """Training manager for SourcePredictorTransformer."""
    
    def __init__(
        self,
        model: SourcePredictorTransformer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        device: str = 'cuda',
        use_amp: bool = True,
        grad_accum_steps: int = 1,
        checkpoint_dir: str = 'checkpoints',
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_amp = use_amp and device == 'cuda'
        self.grad_accum_steps = grad_accum_steps
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,  # Will be updated in train()
            eta_min=1e-6,
        )
        
        # AMP scaler
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        # Tracking
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': [],
            'lr': [],
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_l1 = 0.0
        n_batches = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            Y_noisy = batch['Y_noisy'].to(self.device)
            sensor_xy = batch['sensor_xy'].to(self.device)
            sensor_mask = batch['sensor_mask'].to(self.device)
            kappa = batch['kappa'].to(self.device)
            bc = batch['bc'].to(self.device)
            n_sources = batch['n_sources'].to(self.device)
            sources = batch['sources'].to(self.device)
            source_mask = batch['source_mask'].to(self.device)
            global_features = batch['global_features'].to(self.device)
            
            # Forward pass
            if self.use_amp:
                with autocast('cuda'):
                    pred = self.model(Y_noisy, sensor_xy, sensor_mask, kappa, bc, n_sources, global_features)
                    loss, loss_dict = combined_loss(pred, sources, source_mask)
                    loss = loss / self.grad_accum_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                pred = self.model(Y_noisy, sensor_xy, sensor_mask, kappa, bc, n_sources, global_features)
                loss, loss_dict = combined_loss(pred, sources, source_mask)
                loss = loss / self.grad_accum_steps
                
                loss.backward()
                
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            total_loss += loss_dict['loss']
            total_mse += loss_dict['mse_loss']
            total_l1 += loss_dict['l1_loss']
            n_batches += 1
            
            pbar.set_postfix({
                'loss': f"{total_loss / n_batches:.4f}",
                'mse': f"{total_mse / n_batches:.4f}",
            })
        
        return {
            'loss': total_loss / n_batches,
            'mse_loss': total_mse / n_batches,
            'l1_loss': total_l1 / n_batches,
        }
    
    @torch.no_grad()
    def validate(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        all_preds = []
        all_targets = []
        all_masks = []
        
        for batch in tqdm(self.val_loader, desc='Validating'):
            Y_noisy = batch['Y_noisy'].to(self.device)
            sensor_xy = batch['sensor_xy'].to(self.device)
            sensor_mask = batch['sensor_mask'].to(self.device)
            kappa = batch['kappa'].to(self.device)
            bc = batch['bc'].to(self.device)
            n_sources = batch['n_sources'].to(self.device)
            sources = batch['sources'].to(self.device)
            source_mask = batch['source_mask'].to(self.device)
            global_features = batch['global_features'].to(self.device)
            
            if self.use_amp:
                with autocast('cuda'):
                    pred = self.model(Y_noisy, sensor_xy, sensor_mask, kappa, bc, n_sources, global_features)
            else:
                pred = self.model(Y_noisy, sensor_xy, sensor_mask, kappa, bc, n_sources, global_features)
            
            _, loss_dict = combined_loss(pred, sources, source_mask)
            total_loss += loss_dict['loss']
            n_batches += 1
            
            all_preds.append(pred.cpu())
            all_targets.append(sources.cpu())
            all_masks.append(source_mask.cpu())
        
        # Compute metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        
        metrics = compute_metrics(all_preds, all_targets, all_masks)
        
        return {'loss': total_loss / n_batches}, metrics
    
    def train(
        self,
        epochs: int = 100,
        patience: int = 10,
        save_every: int = 10,
    ) -> Dict:
        """
        Full training loop.
        
        Args:
            epochs: Number of epochs
            patience: Early stopping patience
            save_every: Save checkpoint every N epochs
        
        Returns:
            Training history
        """
        # Update scheduler T_max
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=1e-6,
        )
        
        print(f"\n{'='*60}")
        print(f"Starting training for {epochs} epochs")
        print(f"Device: {self.device}")
        print(f"AMP: {self.use_amp}")
        print(f"Gradient accumulation steps: {self.grad_accum_steps}")
        print(f"{'='*60}\n")
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            # Train
            train_results = self.train_epoch()
            
            # Validate
            val_results, val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Track history
            self.history['train_loss'].append(train_results['loss'])
            self.history['val_loss'].append(val_results['loss'])
            self.history['val_metrics'].append(val_metrics)
            self.history['lr'].append(current_lr)
            
            # Log
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch}/{epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_results['loss']:.4f} (MSE: {train_results['mse_loss']:.4f})")
            print(f"  Val Loss: {val_results['loss']:.4f}")
            print(f"  Val MAE - x: {val_metrics['mae_x']:.4f}, y: {val_metrics['mae_y']:.4f}, q: {val_metrics['mae_q']:.4f}")
            print(f"  Val Position Error: {val_metrics['mae_position']:.4f}")
            print(f"  LR: {current_lr:.2e}")
            
            # Check for improvement
            if val_results['loss'] < self.best_val_loss:
                self.best_val_loss = val_results['loss']
                self.epochs_without_improvement = 0
                self.save_checkpoint('best.pt', epoch, val_results['loss'])
                print(f"  ✓ New best model saved!")
            else:
                self.epochs_without_improvement += 1
                print(f"  No improvement ({self.epochs_without_improvement}/{patience})")
            
            # Save periodic checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(f'epoch_{epoch}.pt', epoch, val_results['loss'])
            
            # Early stopping
            if self.epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        # Save final model and history
        self.save_checkpoint('final.pt', epoch, val_results['loss'])
        self.save_history()
        
        return self.history
    
    def save_checkpoint(self, filename: str, epoch: int, val_loss: float):
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'config': {
                'd_model': self.model.d_model,
                'd_temporal': self.model.d_temporal,
                'd_position': self.model.d_position,
                'd_meta': self.model.d_meta,
                'n_heads': self.model.n_heads,
                'n_encoder_layers': self.model.n_encoder_layers,
                'n_decoder_layers': self.model.n_decoder_layers,
                'dropout': self.model.dropout,
                'K_max': self.model.K_max,
                'Lx': self.model.Lx,
                'Ly': self.model.Ly,
                'q_min': self.model.q_min,
                'q_max': self.model.q_max,
            }
        }, path)
    
    def save_history(self):
        """Save training history to JSON."""
        path = self.checkpoint_dir / 'history.json'
        
        # Convert numpy types to Python types for JSON serialization
        history_serializable = {}
        for key, value in self.history.items():
            if key == 'val_metrics':
                history_serializable[key] = [
                    {k: float(v) for k, v in d.items()} for d in value
                ]
            else:
                history_serializable[key] = [float(v) for v in value]
        
        with open(path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
    
    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        return checkpoint.get('epoch', 0)


def main():
    parser = argparse.ArgumentParser(description='Train SourcePredictorTransformer')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to pre-generated dataset (pickle)')
    parser.add_argument('--num_samples', type=int, default=20000,
                        help='Number of samples to generate if no data_path')
    parser.add_argument('--max_time_steps', type=int, default=200,
                        help='Max time steps (for padding/truncation)')
    
    # Model arguments
    parser.add_argument('--d_model', type=int, default=128,
                        help='Model dimension')
    parser.add_argument('--n_encoder_layers', type=int, default=3,
                        help='Number of encoder layers')
    parser.add_argument('--n_decoder_layers', type=int, default=2,
                        help='Number of decoder layers')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--grad_accum', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable automatic mixed precision')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode (small dataset, few epochs)')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Debug mode adjustments
    if args.debug:
        args.num_samples = 100
        args.epochs = 2
        args.batch_size = 4
        args.num_workers = 0
        print("🐛 Debug mode enabled")
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Load or generate dataset
    if args.data_path and Path(args.data_path).exists():
        print(f"Loading dataset from {args.data_path}")
        dataset = load_dataset(args.data_path)
    else:
        print(f"Generating dataset with {args.num_samples} samples...")
        dataset = generate_dataset(num_samples=args.num_samples, seed=args.seed)
        
        # Optionally save for reuse
        data_dir = Path('data')
        data_dir.mkdir(exist_ok=True)
        save_path = data_dir / f'train_dataset_{args.num_samples}.pkl'
        
        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Dataset saved to {save_path}")
    
    print(f"Dataset size: {len(dataset['samples'])} samples")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        dataset,
        batch_size=args.batch_size,
        val_split=0.1,
        num_workers=args.num_workers,
        max_time_steps=args.max_time_steps,
        seed=args.seed,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    model = SourcePredictorTransformer.from_config({
        'd_model': args.d_model,
        'n_encoder_layers': args.n_encoder_layers,
        'n_decoder_layers': args.n_decoder_layers,
    })
    print(f"\n{model_summary(model)}\n")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        use_amp=not args.no_amp,
        grad_accum_steps=args.grad_accum,
        checkpoint_dir=args.checkpoint_dir,
    )
    
    # Train
    history = trainer.train(
        epochs=args.epochs,
        patience=args.patience,
        save_every=10,
    )
    
    print("\n✅ Training complete!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")


if __name__ == '__main__':
    main()
