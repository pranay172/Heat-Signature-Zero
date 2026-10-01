"""
SourcePredictorTransformer: Neural network for heat source parameter estimation.

Architecture:
1. TemporalEncoder: 1D CNN per sensor → temporal features
2. PositionEncoder: MLP on sensor positions → spatial features  
3. MetadataEncoder: Embed kappa, bc → metadata features
4. SensorTransformer: Self-attention over sensors
5. SourceQueryDecoder: Cross-attention from source queries to sensors
6. OutputHeads: Bounded predictions for (x, y, q)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# ==============================================================================
# Configuration defaults
# ==============================================================================
DEFAULT_CONFIG = {
    'd_model': 128,
    'd_temporal': 64,
    'd_position': 32,
    'd_meta': 16,
    'n_heads': 4,
    'n_encoder_layers': 3,
    'n_decoder_layers': 2,
    'dropout': 0.1,
    'K_max': 3,
    'Lx': 2.0,
    'Ly': 1.0,
    'q_min': 0.5,
    'q_max': 2.0,
}


# ==============================================================================
# Submodules
# ==============================================================================
class TemporalEncoder(nn.Module):
    """
    Encodes a single sensor's time-series using 1D CNN.
    
    Input: (T,) or (B, T)
    Output: (d_temporal,) or (B, d_temporal)
    """
    
    def __init__(self, d_out: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        # Input to proj is 64 * 2 (Avg + Max)
        self.proj = nn.Linear(64 * 2, d_out)
        self.norm = nn.LayerNorm(d_out)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T) time series for one sensor
        Returns:
            (B, d_out) temporal features
        """
        # x: (B, T) -> (B, 1, T)
        x = x.unsqueeze(1)
        
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        
        # Pool over time
        x_avg = self.pool(x).squeeze(-1)  # (B, 64)
        x_max = self.max_pool(x).squeeze(-1)  # (B, 64)
        x = torch.cat([x_avg, x_max], dim=-1)  # (B, 128)
        
        x = self.proj(x)
        x = self.norm(x)
        
        return x


class PositionEncoder(nn.Module):
    """
    Encodes sensor (x, y) position using MLP.
    
    Input: (2,) normalized position
    Output: (d_position,)
    """
    
    def __init__(self, d_out: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.GELU(),
            nn.Linear(32, d_out),
            nn.LayerNorm(d_out),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, M, 2) sensor positions
        Returns:
            (B, M, d_out) position features
        """
        return self.mlp(x)


class MetadataEncoder(nn.Module):
    """
    Encodes sample metadata (kappa, bc) into a feature vector.
    
    Input: kappa (float), bc (int: 0=dirichlet, 1=neumann)
    Output: (d_meta,)
    """
    
    def __init__(self, d_out: int = 16):
        super().__init__()
        self.bc_embed = nn.Embedding(2, 8)  # dirichlet/neumann
        self.mlp = nn.Sequential(
            nn.Linear(8 + 1, 32),  # bc_embed + kappa
            nn.GELU(),
            nn.Linear(32, d_out),
            nn.LayerNorm(d_out),
        )
    
    def forward(self, kappa: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            kappa: (B,) thermal diffusivity
            bc: (B,) boundary condition type (0 or 1)
        Returns:
            (B, d_out) metadata features
        """
        bc_feat = self.bc_embed(bc)  # (B, 8)
        kappa_feat = kappa.unsqueeze(-1)  # (B, 1)
        x = torch.cat([bc_feat, kappa_feat], dim=-1)  # (B, 9)
        return self.mlp(x)  # (B, d_out)


class GlobalStatEncoder(nn.Module):
    """
    Encodes global statistics (max, mean, std, rise_rate).
    """
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        
    def forward(self, global_features: torch.Tensor) -> torch.Tensor:
        # global_features: (B, 4)
        return self.mlp(global_features)


class SensorFusion(nn.Module):
    """
    Fuses temporal, position, and metadata features into unified sensor embeddings.
    """
    
    def __init__(self, d_temporal: int, d_position: int, d_meta: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_temporal + d_position + d_meta, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        h_temporal: torch.Tensor,
        h_position: torch.Tensor,
        h_meta: torch.Tensor,
        h_global: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            h_temporal: (B, M, d_temporal)
            h_position: (B, M, d_position)
            h_meta: (B, d_meta) - broadcast to all sensors
        Returns:
            (B, M, d_model) fused sensor embeddings
        """
        B, M, _ = h_temporal.shape
        
        # Broadcast metadata to all sensors
        h_meta_expanded = h_meta.unsqueeze(1).expand(B, M, -1)  # (B, M, d_meta)
        
        # Concatenate all features
        h = torch.cat([h_temporal, h_position, h_meta_expanded], dim=-1)
        
        # Project and normalize
        h = self.proj(h)
        
        # Add global features if present
        if h_global is not None:
             h = h + h_global.unsqueeze(1)
        
        h = self.norm(h)
        
        return h


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 20):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding (up to sequence length)."""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


# ==============================================================================
# Main Model
# ==============================================================================
class SourcePredictorTransformer(nn.Module):
    """
    Transformer-based model for predicting heat source parameters.
    
    Takes sensor readings and positions, outputs source (x, y, q) predictions.
    Uses cross-attention from learnable source queries to sensor embeddings.
    """
    
    def __init__(
        self,
        d_model: int = 128,
        d_temporal: int = 64,
        d_position: int = 32,
        d_meta: int = 16,
        n_heads: int = 4,
        n_encoder_layers: int = 3,
        n_decoder_layers: int = 2,
        dropout: float = 0.1,
        K_max: int = 3,
        Lx: float = 2.0,
        Ly: float = 1.0,
        q_min: float = 0.5,
        q_max: float = 2.0,
    ):
        super().__init__()
        
        # Store ALL config parameters for proper checkpoint restoration
        self.d_model = d_model
        self.d_temporal = d_temporal
        self.d_position = d_position
        self.d_meta = d_meta
        self.n_heads = n_heads
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.dropout = dropout
        self.K_max = K_max
        self.Lx = Lx
        self.Ly = Ly
        self.q_min = q_min
        self.q_max = q_max
        
        # Encoders
        self.temporal_encoder = TemporalEncoder(d_out=d_temporal)
        self.position_encoder = PositionEncoder(d_out=d_position)
        self.metadata_encoder = MetadataEncoder(d_out=d_meta)
        self.global_encoder = GlobalStatEncoder(d_model=d_model)
        
        self.sensor_fusion = SensorFusion(d_temporal, d_position, d_meta, d_model)
        
        # Sensor Transformer (self-attention over sensors)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.sensor_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_encoder_layers,
        )
        
        # Learnable source queries
        self.source_queries = nn.Parameter(torch.randn(K_max, d_model) * 0.02)
        
        # Source Query Decoder (cross-attention to sensors)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.source_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=n_decoder_layers,
        )
        
        # Output heads - separate MLPs for x, y, q
        self.head_x = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.head_y = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.head_q = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode_sensors(
        self,
        Y_noisy: torch.Tensor,
        sensor_xy: torch.Tensor,
        sensor_mask: torch.Tensor,
        kappa: torch.Tensor,
        bc: torch.Tensor,
        global_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode sensor data into embeddings.
        
        Args:
            Y_noisy: (B, T, M) sensor readings
            sensor_xy: (B, M, 2) sensor positions
            sensor_mask: (B, M) valid sensor mask
            kappa: (B,) thermal diffusivity
            bc: (B,) boundary condition
        
        Returns:
            (B, M, d_model) sensor embeddings
        """
        B, T, M = Y_noisy.shape
        
        # Encode each sensor's time-series
        # Reshape: (B, T, M) -> (B*M, T)
        Y_flat = Y_noisy.permute(0, 2, 1).reshape(B * M, T)
        h_temporal_flat = self.temporal_encoder(Y_flat)  # (B*M, d_temporal)
        h_temporal = h_temporal_flat.view(B, M, -1)  # (B, M, d_temporal)
        
        # Encode sensor positions
        h_position = self.position_encoder(sensor_xy)  # (B, M, d_position)
        
        # Encode metadata
        h_meta = self.metadata_encoder(kappa, bc)  # (B, d_meta)
        
        # Encode global features
        if global_features is not None:
             h_global = self.global_encoder(global_features) # (B, d_model)
        else:
             h_global = None
        
        # Fuse all features
        h_sensor = self.sensor_fusion(h_temporal, h_position, h_meta, h_global)  # (B, M, d_model)
        
        # Self-attention over sensors
        # Create attention mask (True = masked, so we invert)
        attn_mask = ~sensor_mask  # (B, M)
        h_sensor = self.sensor_transformer(
            h_sensor,
            src_key_padding_mask=attn_mask,
        )
        
        return h_sensor
    
    def decode_sources(
        self,
        h_sensor: torch.Tensor,
        sensor_mask: torch.Tensor,
        n_sources: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode source parameters using cross-attention.
        
        Args:
            h_sensor: (B, M, d_model) sensor embeddings
            sensor_mask: (B, M) valid sensor mask
            n_sources: (B,) number of sources per sample
        
        Returns:
            (B, K_max, 3) source parameters (x, y, q), padded to K_max
        """
        B = h_sensor.size(0)
        device = h_sensor.device
        
        # Expand source queries for batch
        queries = self.source_queries.unsqueeze(0).expand(B, -1, -1)  # (B, K_max, d_model)
        
        # Cross-attention from queries to sensors
        attn_mask = ~sensor_mask  # (B, M)
        h_sources = self.source_decoder(
            queries,
            h_sensor,
            memory_key_padding_mask=attn_mask,
        )  # (B, K_max, d_model)
        
        # Decode to (x, y, q) with sigmoid for bounded output
        x = torch.sigmoid(self.head_x(h_sources).squeeze(-1))  # (B, K_max) in [0, 1]
        y = torch.sigmoid(self.head_y(h_sources).squeeze(-1))  # (B, K_max) in [0, 1]
        q = torch.sigmoid(self.head_q(h_sources).squeeze(-1))  # (B, K_max) in [0, 1]
        
        # Stack into (B, K_max, 3)
        sources = torch.stack([x, y, q], dim=-1)
        
        return sources
    
    def forward(
        self,
        Y_noisy: torch.Tensor,
        sensor_xy: torch.Tensor,
        sensor_mask: torch.Tensor,
        kappa: torch.Tensor,
        bc: torch.Tensor,
        n_sources: Optional[torch.Tensor] = None,
        global_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            Y_noisy: (B, T, M) sensor readings (normalized)
            sensor_xy: (B, M, 2) sensor positions (normalized to [0, 1])
            sensor_mask: (B, M) valid sensor mask
            kappa: (B,) thermal diffusivity
            bc: (B,) boundary condition (0=dirichlet, 1=neumann)
            n_sources: (B,) number of sources per sample (optional, for masking output)
        
        Returns:
            (B, K_max, 3) source parameters in NORMALIZED space [0, 1]
            Use denormalize_sources() to get actual (x, y, q) values
        """
        # Encode sensors
        h_sensor = self.encode_sensors(Y_noisy, sensor_xy, sensor_mask, kappa, bc, global_features)
        
        # Decode sources
        sources = self.decode_sources(h_sensor, sensor_mask, n_sources)
        
        return sources
    
    def denormalize_sources(self, sources: torch.Tensor) -> torch.Tensor:
        """
        Convert normalized source parameters to original scale.
        
        Args:
            sources: (B, K, 3) with values in [0, 1]
        
        Returns:
            (B, K, 3) with x in [0, Lx], y in [0, Ly], q in [q_min, q_max]
        """
        x = sources[..., 0] * self.Lx
        y = sources[..., 1] * self.Ly
        q = sources[..., 2] * (self.q_max - self.q_min) + self.q_min
        return torch.stack([x, y, q], dim=-1)
    
    @classmethod
    def from_config(cls, config: dict = None):
        """Create model from config dict."""
        cfg = DEFAULT_CONFIG.copy()
        if config:
            cfg.update(config)
        return cls(**cfg)
    
    def save(self, path: str):
        """Save model checkpoint with complete config."""
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'd_model': self.d_model,
                'd_temporal': self.d_temporal,
                'd_position': self.d_position,
                'd_meta': self.d_meta,
                'n_heads': self.n_heads,
                'n_encoder_layers': self.n_encoder_layers,
                'n_decoder_layers': self.n_decoder_layers,
                'dropout': self.dropout,
                'K_max': self.K_max,
                'Lx': self.Lx,
                'Ly': self.Ly,
                'q_min': self.q_min,
                'q_max': self.q_max,
            }
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu'):
        """Load model from checkpoint.
        
        Handles both formats:
        - Model's save() format: {'state_dict': ..., 'config': ...}
        - Trainer's format: {'model_state_dict': ..., 'epoch': ...}
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        # Get config (may not exist in Trainer checkpoints)
        config = checkpoint.get('config', {})
        model = cls.from_config(config)
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Assume the checkpoint IS the state dict
            model.load_state_dict(checkpoint)
        
        return model


# ==============================================================================
# Utility functions
# ==============================================================================
def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: SourcePredictorTransformer) -> str:
    """Get model summary string."""
    lines = [
        "SourcePredictorTransformer",
        "=" * 50,
        f"d_model: {model.d_model}",
        f"K_max: {model.K_max}",
        f"Total parameters: {count_parameters(model):,}",
        "",
        "Submodules:",
    ]
    
    for name, module in model.named_children():
        n_params = sum(p.numel() for p in module.parameters())
        lines.append(f"  {name}: {n_params:,} params")
    
    return "\n".join(lines)


# ==============================================================================
# Test
# ==============================================================================
if __name__ == '__main__':
    # Test the model
    print("Testing SourcePredictorTransformer...")
    
    model = SourcePredictorTransformer.from_config()
    print(model_summary(model))
    
    # Create dummy batch
    B, T, M = 4, 200, 5
    Y_noisy = torch.randn(B, T, M)
    sensor_xy = torch.rand(B, M, 2)
    sensor_mask = torch.ones(B, M, dtype=torch.bool)
    sensor_mask[0, 4] = False  # Mask one sensor in first sample
    kappa = torch.tensor([0.05, 0.1, 0.05, 0.1])
    bc = torch.tensor([0, 1, 0, 1])
    n_sources = torch.tensor([1, 2, 1, 2])
    
    # Forward pass
    with torch.no_grad():
        output = model(Y_noisy, sensor_xy, sensor_mask, kappa, bc, n_sources)
    
    print(f"\nInput shapes:")
    print(f"  Y_noisy: {Y_noisy.shape}")
    print(f"  sensor_xy: {sensor_xy.shape}")
    print(f"  sensor_mask: {sensor_mask.shape}")
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Output (normalized, sample 0): {output[0]}")
    
    # Denormalize
    output_denorm = model.denormalize_sources(output)
    print(f"Output (denormalized, sample 0): {output_denorm[0]}")
    
    print("\n✅ Model test passed!")