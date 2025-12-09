"""
Dual-Head HYBRID+FUSED ViTA-Seg Model

This model combines the best of both approaches:
1. Shared early decoder layers (common low-level features)
2. Task-specific late decoder layers (specialized high-level features)
3. Fusion of occluded penultimate features into amodal prediction

Architecture Innovation:
- Hybrid: Task-specific branches learn specialized features
- Fusion: Occluded specialized features help refine amodal prediction
- Result: Best of both worlds - specialization + collaboration

The hypothesis: Task-specific learning paths + cross-task fusion should
yield the best performance by combining specialized expertise with collaboration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Optional
import warnings

try:
    from model import AmodalViT as BaseAmodalViT
    BASE_MODEL_AVAILABLE = True
except ImportError:
    BASE_MODEL_AVAILABLE = False
    warnings.warn("Base model not available. Make sure model.py is in the same directory.")


class HybridFusedDualHeadDecoder(nn.Module):
    """
    Hybrid+Fused dual-head decoder.
    
    Architecture:
        Shared early decoder (768→512→256) - Common low-level features
        ↓
        Split into task-specific branches
        ↓
        Occluded late decoder (256→128→64) - Occluded-specific features
        Amodal late decoder (256→128→64) - Amodal-specific features
        ↓
        Fusion: Occluded penultimate + Amodal penultimate → Final amodal mask
    """
    
    def __init__(
        self,
        encoder_dim: int = 768,
        shared_channels: list = [512, 256],  # Early shared layers
        task_specific_channels: list = [128, 64],  # Late task-specific layers
        num_patches: int = 196,
        img_size: int = 224,
        patch_size: int = 16
    ):
        super().__init__()
        
        self.num_patches = num_patches
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        
        # === SHARED EARLY DECODER LAYERS ===
        # Learn common low-level features (edges, textures, basic shapes)
        self.shared_early_layers = nn.ModuleList()
        
        in_channels = encoder_dim
        for out_channels in shared_channels:
            self.shared_early_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                )
            )
            in_channels = out_channels
        
        # === TASK-SPECIFIC LATE DECODER LAYERS ===
        # Occluded branch - learns to detect hidden regions through specialized path
        self.occluded_late_layers = nn.ModuleList()
        in_channels = shared_channels[-1]
        for out_channels in task_specific_channels:
            self.occluded_late_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                )
            )
            in_channels = out_channels
        
        # Amodal branch - learns complete object shape through specialized path
        self.amodal_late_layers = nn.ModuleList()
        in_channels = shared_channels[-1]
        for out_channels in task_specific_channels:
            self.amodal_late_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                )
            )
            in_channels = out_channels
        
        # === PENULTIMATE LAYERS (before fusion) ===
        final_channels = task_specific_channels[-1]
        
        # Occluded penultimate (produces features for fusion)
        self.occluded_penultimate = nn.Sequential(
            nn.Conv2d(final_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Amodal penultimate (produces features for fusion)
        self.amodal_penultimate = nn.Sequential(
            nn.Conv2d(final_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # === FUSION LAYER ===
        # Combines specialized occluded and amodal features
        # Input: 32 (amodal) + 32 (occluded) = 64 channels
        self.fusion = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # === FINAL PREDICTION HEADS ===
        # Occluded final (independent)
        self.occluded_final = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Amodal final (from fused features)
        self.amodal_final = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            features: Encoder output [B, num_patches+1, embed_dim] (with CLS token)
            
        Returns:
            dict with keys:
                - 'amodal': Fused amodal mask [B, 1, H, W]
                - 'occluded': Occluded mask [B, 1, H, W]
                - 'amodal_penultimate': Amodal penultimate features [B, 32, H, W]
                - 'occluded_penultimate': Occluded penultimate features [B, 32, H, W]
        """
        B = features.size(0)
        
        # Remove CLS token
        features = features[:, 1:, :]  # [B, num_patches, embed_dim]
        
        # Reshape to spatial feature map
        features = features.transpose(1, 2)  # [B, embed_dim, num_patches]
        features = features.reshape(B, -1, self.grid_size, self.grid_size)
        
        # === SHARED EARLY DECODER ===
        # Both tasks benefit from common low-level features
        x = features
        for layer in self.shared_early_layers:
            x = layer(x)
        # x is now [B, 256, 56, 56] - shared features at mid-resolution
        
        # === SPLIT INTO TASK-SPECIFIC BRANCHES ===
        # Occluded branch - specialized path for hidden region detection
        x_occluded = x
        for layer in self.occluded_late_layers:
            x_occluded = layer(x_occluded)
        # x_occluded is now [B, 64, 224, 224] - specialized occluded features
        
        # Amodal branch - specialized path for complete shape reasoning
        x_amodal = x
        for layer in self.amodal_late_layers:
            x_amodal = layer(x_amodal)
        # x_amodal is now [B, 64, 224, 224] - specialized amodal features
        
        # === PENULTIMATE FEATURES ===
        occluded_penult = self.occluded_penultimate(x_occluded)  # [B, 32, 224, 224]
        amodal_penult = self.amodal_penultimate(x_amodal)        # [B, 32, 224, 224]
        
        # === OCCLUDED BRANCH (INDEPENDENT) ===
        occluded_mask = self.occluded_final(occluded_penult)  # [B, 1, 224, 224]
        
        # === AMODAL BRANCH (FUSED) ===
        # FUSION: Concatenate specialized features from both branches
        fused = torch.cat([amodal_penult, occluded_penult], dim=1)  # [B, 64, 224, 224]
        fused = self.fusion(fused)  # [B, 32, 224, 224]
        
        # Final amodal prediction from fused specialized features
        amodal_mask = self.amodal_final(fused)  # [B, 1, 224, 224]
        
        return {
            'amodal': amodal_mask,
            'occluded': occluded_mask,
            'amodal_penultimate': amodal_penult,
            'occluded_penultimate': occluded_penult
        }


class HybridFusedDualHeadAmodalViT(nn.Module):
    """
    Complete hybrid+fused dual-head ViTA-Seg model.
    
    Combines hybrid architecture (task-specific branches) with fusion (cross-task collaboration).
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 4,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        shared_channels: list = [512, 256],
        task_specific_channels: list = [128, 64],
        pretrained_model: str = None,
        freeze_pretrained: bool = False,
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        if not BASE_MODEL_AVAILABLE:
            raise ImportError("Base model not available. Make sure model.py is accessible.")
        
        # Create base model to reuse encoder with pretrained support
        self.base_model = BaseAmodalViT(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=4,
            embed_dim=embed_dim,
            num_layers=depth,
            num_heads=num_heads,
            mlp_ratio=int(mlp_ratio),
            dropout=dropout,
            num_decoder_layers=4,
            pretrained_model=pretrained_model,
            freeze_pretrained=freeze_pretrained
        )
        
        # Reuse encoder from base model
        self.encoder = self.base_model.encoder
        
        # Store pretrained info
        self.pretrained_model = pretrained_model
        
        num_patches = (img_size // patch_size) ** 2
        
        # Create hybrid+fused dual-head decoder
        self.decoder = HybridFusedDualHeadDecoder(
            encoder_dim=embed_dim,
            shared_channels=shared_channels,
            task_specific_channels=task_specific_channels,
            num_patches=num_patches,
            img_size=img_size,
            patch_size=patch_size
        )
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_encoder()
    
    def _freeze_encoder(self):
        """Freeze encoder parameters"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("✅ Encoder frozen")
    
    def forward(self, image: Tensor, visible_mask: Tensor) -> Dict[str, Tensor]:
        """
        Forward pass through the hybrid+fused dual-head model.
        
        Args:
            image: RGB image [B, 3, H, W]
            visible_mask: Visible mask [B, 1, H, W]
            
        Returns:
            dict with keys:
                - 'amodal': Fused amodal mask [B, 1, H, W]
                - 'occluded': Occluded mask [B, 1, H, W]
                - 'amodal_penultimate': Amodal penultimate features
                - 'occluded_penultimate': Occluded penultimate features
        """
        # Concatenate RGB and visible mask
        x = torch.cat([image, visible_mask], dim=1)  # [B, 4, H, W]
        
        # Pass through encoder
        encoded_features, skip_features = self.encoder(x)
        
        # Hybrid+fused dual-head decoder
        predictions = self.decoder(encoded_features)
        
        return predictions
    
    def load_pretrained_encoder(self, checkpoint_path: str):
        """Load pretrained encoder weights"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Filter encoder weights
        encoder_dict = {}
        for k, v in state_dict.items():
            if any(key in k for key in ['patch_embed', 'cls_token', 'pos_embed', 'encoder_blocks', 'encoder_norm']):
                encoder_dict[k] = v
        
        # Load encoder weights
        missing, unexpected = self.load_state_dict(encoder_dict, strict=False)
        
        print(f"✅ Loaded pretrained encoder from {checkpoint_path}")
        print(f"   Missing keys (expected - decoder): {len(missing)}")
        print(f"   Unexpected keys: {len(unexpected)}")
    
    def get_trainable_params(self) -> Dict[str, int]:
        """Get count of trainable parameters by component"""
        encoder_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        decoder_params = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        
        # Break down decoder params
        shared_params = sum(p.numel() for layer in self.decoder.shared_early_layers for p in layer.parameters())
        occluded_params = sum(p.numel() for layer in self.decoder.occluded_late_layers for p in layer.parameters())
        occluded_params += sum(p.numel() for p in self.decoder.occluded_penultimate.parameters())
        occluded_params += sum(p.numel() for p in self.decoder.occluded_final.parameters())
        amodal_params = sum(p.numel() for layer in self.decoder.amodal_late_layers for p in layer.parameters())
        amodal_params += sum(p.numel() for p in self.decoder.amodal_penultimate.parameters())
        fusion_params = sum(p.numel() for p in self.decoder.fusion.parameters())
        fusion_params += sum(p.numel() for p in self.decoder.amodal_final.parameters())
        
        return {
            'encoder': encoder_params,
            'decoder_shared_early': shared_params,
            'decoder_occluded_branch': occluded_params,
            'decoder_amodal_branch': amodal_params,
            'decoder_fusion': fusion_params,
            'decoder_total': decoder_params,
            'total': encoder_params + decoder_params
        }


class HybridFusedDualHeadLoss(nn.Module):
    """
    Loss function for hybrid+fused dual-head model.
    
    Same as standard dual-head loss.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        super().__init__()
        
        # Default weights
        default_weights = {
            'amodal': 1.0,
            'occluded': 1.5,
            'dice': 1.0,
            'bce': 1.0
        }
        
        self.weights = weights if weights is not None else default_weights
        
        self.bce = nn.BCELoss()
    
    def dice_loss(self, pred: Tensor, target: Tensor, smooth: float = 1e-6) -> Tensor:
        """Compute Dice loss"""
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        
        return 1 - dice
    
    def forward(
        self,
        predictions: Dict[str, Tensor],
        amodal_target: Tensor,
        visible_target: Tensor
    ) -> Dict[str, Tensor]:
        """
        Compute hybrid+fused dual-head loss.
        
        Args:
            predictions: Dict with 'amodal' and 'occluded' predictions
            amodal_target: Ground truth amodal mask [B, 1, H, W]
            visible_target: Ground truth visible mask [B, 1, H, W]
            
        Returns:
            Dict with individual losses and total loss
        """
        pred_amodal = predictions['amodal']
        pred_occluded = predictions['occluded']
        
        # Compute occluded target
        amodal_bool = amodal_target > 0.5
        visible_bool = visible_target < 0.5
        occluded_target = (amodal_bool & visible_bool).float()
        
        # Amodal losses (fused prediction)
        amodal_bce = self.bce(pred_amodal, amodal_target)
        amodal_dice = self.dice_loss(pred_amodal, amodal_target)
        
        # Occluded losses
        occluded_bce = self.bce(pred_occluded, occluded_target)
        occluded_dice = self.dice_loss(pred_occluded, occluded_target)
        
        # Weighted combination
        amodal_loss = (
            self.weights['bce'] * amodal_bce +
            self.weights['dice'] * amodal_dice
        )
        
        occluded_loss = (
            self.weights['bce'] * occluded_bce +
            self.weights['dice'] * occluded_dice
        )
        
        total_loss = (
            self.weights['amodal'] * amodal_loss +
            self.weights['occluded'] * occluded_loss
        )
        
        return {
            'total': total_loss,
            'amodal': amodal_loss,
            'occluded': occluded_loss,
            'amodal_bce': amodal_bce,
            'amodal_dice': amodal_dice,
            'occluded_bce': occluded_bce,
            'occluded_dice': occluded_dice
        }


if __name__ == "__main__":
    # Test the hybrid+fused dual-head model
    print("Testing Hybrid+Fused Dual-Head ViTA-Seg Model...")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model
    model = HybridFusedDualHeadAmodalViT(
        img_size=224,
        patch_size=16,
        in_channels=4,
        embed_dim=768,
        depth=12,
        num_heads=12,
        dropout=0.1,
        shared_channels=[512, 256],
        task_specific_channels=[128, 64],
        pretrained_model=None
    ).to(device)
    
    # Count parameters
    params = model.get_trainable_params()
    print(f"\nTrainable parameters:")
    for name, count in params.items():
        print(f"  {name}: {count:,}")
    
    # Test forward pass
    batch_size = 2
    image = torch.randn(batch_size, 3, 224, 224).to(device)
    visible_mask = torch.randn(batch_size, 1, 224, 224).sigmoid().to(device)
    amodal_mask = torch.randn(batch_size, 1, 224, 224).sigmoid().to(device)
    
    print(f"\nInput shapes:")
    print(f"  Image: {image.shape}")
    print(f"  Visible mask: {visible_mask.shape}")
    print(f"  Amodal mask (target): {amodal_mask.shape}")
    
    # Forward pass
    predictions = model(image, visible_mask)
    
    print(f"\nOutput shapes:")
    print(f"  Amodal (fused): {predictions['amodal'].shape}")
    print(f"  Occluded: {predictions['occluded'].shape}")
    print(f"  Amodal penultimate: {predictions['amodal_penultimate'].shape}")
    print(f"  Occluded penultimate: {predictions['occluded_penultimate'].shape}")
    
    # Test loss
    criterion = HybridFusedDualHeadLoss()
    losses = criterion(predictions, amodal_mask, visible_mask)
    
    print(f"\nLosses:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")
    
    print("\n" + "=" * 70)
    print("Key Architecture Features:")
    print("  1. Shared early decoder (768→512→256) - Common low-level features")
    print("  2. Split into task-specific branches at 256 channels")
    print("  3. Occluded late decoder (256→128→64) - Specialized occluded features")
    print("  4. Amodal late decoder (256→128→64) - Specialized amodal features")
    print("  5. FUSION: Occluded penultimate + Amodal penultimate → Final amodal")
    print("  6. Best of both worlds: Specialization + Collaboration!")
    print("=" * 70)
    
    print("\n✅ Hybrid+Fused dual-head model test completed successfully!")
