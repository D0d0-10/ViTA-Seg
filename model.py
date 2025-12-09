import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from typing import Optional, Tuple, List, Dict, Any
import warnings

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    warnings.warn("timm not available. Install with: pip install timm")

try:
    from transformers import ViTModel, ViTConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("transformers not available. Install with: pip install transformers")


class PatchEmbedding(nn.Module):
    """
    Converte l'input (RGB + Visible Mask) in patch embeddings
    Supporta l'adattamento di embedding pre-addestrati da 3 a 4 canali
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 4, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.in_channels = in_channels
        
        # Proiezione lineare delle patch
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def load_pretrained_projection(self, pretrained_weight: Tensor):
        """
        Carica pesi pre-addestrati e li adatta da 3 a 4 canali
        
        Args:
            pretrained_weight: Tensor di shape [embed_dim, 3, patch_size, patch_size]
        """
        if pretrained_weight.size(1) != 3:
            raise ValueError(f"Expected pretrained weight with 3 input channels, got {pretrained_weight.size(1)}")
            
        # Crea nuovo peso con 4 canali
        new_weight = torch.zeros(pretrained_weight.size(0), 4, pretrained_weight.size(2), pretrained_weight.size(3))
        
        # Copia i primi 3 canali (RGB)
        new_weight[:, :3, :, :] = pretrained_weight
        
        # Inizializza il 4° canale (mask) con media dei canali RGB o zeros
        new_weight[:, 3, :, :] = pretrained_weight.mean(dim=1)  # Media RGB
        
        # Assegna i pesi
        self.projection.weight.data = new_weight
        
        print(f"✅ Adapted pretrained patch embedding from 3 to 4 channels")
        
    def forward(self, x: Tensor) -> Tensor:
        # x: [B, 4, H, W] (RGB + visible mask)
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, f"Input size {H}x{W} doesn't match expected {self.img_size}x{self.img_size}"
        
        # Proiezione in patch embeddings
        x = self.projection(x)  # [B, embed_dim, H//patch_size, W//patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism
    """
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        
        # Genera Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class MLP(nn.Module):
    """
    Multi-Layer Perceptron block
    """
    def __init__(self, embed_dim: int = 768, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden_dim = embed_dim * mlp_ratio
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer encoder block
    """
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        
    def forward(self, x: Tensor) -> Tensor:
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """
    Vision Transformer Encoder con supporto per modelli pre-addestrati
    """
    def __init__(self, 
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 4,
                 embed_dim: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 pretrained_model: Optional[str] = None,
                 freeze_pretrained: bool = False):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.pretrained_model = pretrained_model
        self.freeze_pretrained = freeze_pretrained
        
        # Check if pretrained model is available
        if pretrained_model and not self._is_pretrained_available():
            warnings.warn(f"Pretrained model {pretrained_model} not available. Training from scratch.")
            pretrained_model = None
        
        # Patch embedding (sempre custom per 4 canali)
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embedding.num_patches
        
        # Initialize model components
        if pretrained_model:
            self._load_pretrained_encoder(pretrained_model)
        else:
            self._init_from_scratch(num_layers, num_heads, mlp_ratio, dropout)
            
        # Freeze pretrained weights if requested
        if freeze_pretrained and self.pretrained_model:
            self._freeze_pretrained_layers()
            
    def _is_pretrained_available(self) -> bool:
        """Controlla se le librerie per modelli pre-addestrati sono disponibili"""
        return TIMM_AVAILABLE or TRANSFORMERS_AVAILABLE
        
    def _init_from_scratch(self, num_layers: int, num_heads: int, mlp_ratio: int, dropout: float):
        """Inizializza il modello da zero"""
        # Positional embeddings with proper initialization and smaller std for stability
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.01)  # Reduced from 0.02 for stability
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.01)  # Reduced from 0.02 for stability
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        print(f"✅ Initialized ViT encoder from scratch with {num_layers} layers (stabilized initialization)")
        print(f"   Position embedding std: 0.01, CLS token std: 0.01")
        
    def _load_pretrained_encoder(self, model_name: str):
        """Carica encoder pre-addestrato da varie fonti"""
        print(f"🔄 Loading pretrained ViT: {model_name}")
        
        if model_name.startswith('timm:') and TIMM_AVAILABLE:
            self._load_from_timm(model_name.replace('timm:', ''))
        elif model_name.startswith('hf:') and TRANSFORMERS_AVAILABLE:
            self._load_from_huggingface(model_name.replace('hf:', ''))
        else:
            # Try timm first, then huggingface
            if TIMM_AVAILABLE:
                try:
                    self._load_from_timm(model_name)
                except:
                    if TRANSFORMERS_AVAILABLE:
                        self._load_from_huggingface(model_name)
                    else:
                        raise ValueError(f"Cannot load {model_name}. Install timm or transformers.")
            elif TRANSFORMERS_AVAILABLE:
                self._load_from_huggingface(model_name)
            else:
                raise ValueError("Neither timm nor transformers available for pretrained models.")
                
    def _load_from_timm(self, model_name: str):
        """Carica modello da timm"""
        try:
            # Carica modello timm
            pretrained_model = timm.create_model(model_name, pretrained=True)
            
            # Estrai componenti
            if hasattr(pretrained_model, 'patch_embed'):
                # Adatta patch embedding
                pretrained_patch_weight = pretrained_model.patch_embed.proj.weight
                self.patch_embedding.load_pretrained_projection(pretrained_patch_weight)
                
                # Copia positional embeddings
                if hasattr(pretrained_model, 'pos_embed'):
                    self.pos_embedding = nn.Parameter(pretrained_model.pos_embed.clone())
                    
                # Copia cls token
                if hasattr(pretrained_model, 'cls_token'):
                    self.cls_token = nn.Parameter(pretrained_model.cls_token.clone())
                    
                # Copia transformer blocks
                if hasattr(pretrained_model, 'blocks'):
                    self.blocks = pretrained_model.blocks
                    
                # Copia layer norm
                if hasattr(pretrained_model, 'norm'):
                    self.norm = pretrained_model.norm
                
                # Initialize dropout (might not be present in pretrained model)
                if hasattr(pretrained_model, 'pos_drop'):
                    self.dropout = pretrained_model.pos_drop
                else:
                    self.dropout = nn.Dropout(0.1)  # Default dropout
                    
            print(f"✅ Loaded pretrained ViT from timm: {model_name}")
            
        except Exception as e:
            raise ValueError(f"Failed to load timm model {model_name}: {e}")
            
    def _load_from_huggingface(self, model_name: str):
        """Carica modello da Hugging Face"""
        try:
            # Carica modello HF
            pretrained_model = ViTModel.from_pretrained(model_name)
            
            # Estrai patch embedding
            pretrained_patch_weight = pretrained_model.embeddings.patch_embeddings.projection.weight
            self.patch_embedding.load_pretrained_projection(pretrained_patch_weight)
            
            # Copia positional embeddings
            self.pos_embedding = nn.Parameter(pretrained_model.embeddings.position_embeddings.clone())
            
            # Copia cls token
            self.cls_token = nn.Parameter(pretrained_model.embeddings.cls_token.clone())
            
            # Copia encoder layers
            self.blocks = pretrained_model.encoder.layer
            
            # Copia layer norm
            self.norm = pretrained_model.layernorm
            
            # Dropout
            self.dropout = nn.Dropout(pretrained_model.config.hidden_dropout_prob)
            
            print(f"✅ Loaded pretrained ViT from Hugging Face: {model_name}")
            
        except Exception as e:
            raise ValueError(f"Failed to load HuggingFace model {model_name}: {e}")
            
    def _freeze_pretrained_layers(self):
        """Congela i layer pre-addestrati"""
        frozen_params = 0
        total_params = 0
        
        for name, param in self.named_parameters():
            total_params += param.numel()
            # Non congelare patch embedding (è stato adattato) e positional embeddings
            if 'patch_embedding' not in name and 'pos_embedding' not in name:
                param.requires_grad = False
                frozen_params += param.numel()
                
        print(f"🔒 Frozen {frozen_params:,}/{total_params:,} parameters ({frozen_params/total_params*100:.1f}%)")
        
    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embedding(x)  # [B, num_patches, embed_dim]
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches+1, embed_dim]
        
        # Add positional embedding
        x = x + self.pos_embedding
        x = self.dropout(x)
        
        # Store intermediate features for skip connections
        features = []
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
            features.append(x)
            
        x = self.norm(x)
        
        return x, features


class DecoderBlock(nn.Module):
    """
    Decoder block with upsampling and skip connections
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        
        # Use a more flexible upsampling approach
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: Tensor, skip: Optional[Tensor] = None) -> Tensor:
        # Upsample to output channels
        x = self.upsample(x)
        
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        
        return x


class AmodalSegmentationDecoder(nn.Module):
    """
    Decoder per la predizione della maschera amodale
    """
    def __init__(self, 
                 embed_dim: int = 768,
                 patch_size: int = 16,
                 img_size: int = 224,
                 num_decoder_layers: int = 4):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        
        # Dimensioni delle feature maps
        feature_size = img_size // patch_size  # 14x14 per 224x224 con patch 16x16
        
        # Proiezione iniziale
        self.initial_projection = nn.Linear(embed_dim, embed_dim)
        
        # Reshape delle features da sequenza a feature map
        self.feature_size = feature_size
        
        # Decoder blocks - Fixed channel progression
        decoder_channels = [embed_dim, 512, 256, 128, 64]
        self.decoders = nn.ModuleList()
        
        for i in range(min(num_decoder_layers, len(decoder_channels) - 1)):
            in_ch = decoder_channels[i]
            out_ch = decoder_channels[i + 1]
            # No skip connections for now
            skip_ch = 0
            self.decoders.append(DecoderBlock(in_ch, skip_ch, out_ch))
            
        # Final prediction head
        final_channels = decoder_channels[min(num_decoder_layers, len(decoder_channels) - 1)]
        self.final_conv = nn.Conv2d(final_channels, 1, kernel_size=1)
        
    def forward(self, encoded_features: Tensor, skip_features: List[Tensor]) -> Tensor:
        # encoded_features: [B, num_patches+1, embed_dim]
        # Remove the cls token
        x = encoded_features[:, 1:, :]  # [B, num_patches, embed_dim]
        
        # Projection
        x = self.initial_projection(x)
        
        # Reshape to feature map
        B = x.shape[0]
        x = x.transpose(1, 2).reshape(B, self.embed_dim, self.feature_size, self.feature_size)
        
        # Pass through decoder blocks
        for i, decoder in enumerate(self.decoders):
            # For now, don't use skip connections from encoder
            # In a more advanced version, you could process skip_features
            x = decoder(x, skip=None)
            
        # Final prediction
        x = self.final_conv(x)  # [B, 1, H, W]
        
        # Upsample to original size if necessary
        if x.size(-1) != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
            
        return torch.sigmoid(x)  # Output in [0, 1]


class AmodalViT(nn.Module):
    """
    Modello completo per Amodal Segmentation con Vision Transformer
    Supporta modelli pre-addestrati e training da zero
    """
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 4,  # RGB + visible mask
                 embed_dim: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 num_decoder_layers: int = 4,
                 pretrained_model: Optional[str] = None,
                 freeze_pretrained: bool = False):
        super().__init__()
        
        self.encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            pretrained_model=pretrained_model,
            freeze_pretrained=freeze_pretrained
        )
        
        self.decoder = AmodalSegmentationDecoder(
            embed_dim=embed_dim,
            patch_size=patch_size,
            img_size=img_size,
            num_decoder_layers=num_decoder_layers
        )
        
        # Inizializza solo il decoder se usando pretrained encoder
        if not pretrained_model:
            self._init_weights()
        else:
            self._init_decoder_weights()
        
    def _init_weights(self):
        """
        Inizializzazione completa dei pesi (training da zero) con stabilità numerica
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Truncated normal initialization for stability
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                torch.nn.init.constant_(m.bias, 0)
                torch.nn.init.constant_(m.weight, 1.0)
        
        print("✅ Applied comprehensive weight initialization for numerical stability")
        
    def _init_decoder_weights(self):
        """
        Inizializza solo i pesi del decoder (quando si usa encoder pre-addestrato)
        """
        for m in self.decoder.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)
                
        print("✅ Initialized decoder weights (encoder is pretrained)")
                
    def forward(self, rgb_image: Tensor, visible_mask: Tensor) -> Tensor:
        """
        Forward pass
        
        Args:
            rgb_image: [B, 3, H, W] - Immagine RGB
            visible_mask: [B, 1, H, W] - Maschera della parte visibile
            
        Returns:
            amodal_mask: [B, 1, H, W] - Maschera amodale predetta
        """
        # Concatena RGB e visible mask
        x = torch.cat([rgb_image, visible_mask], dim=1)  # [B, 4, H, W]
        
        # Encoding
        encoded_features, skip_features = self.encoder(x)
        
        # Decoding
        amodal_mask = self.decoder(encoded_features, skip_features)
        
        return amodal_mask
    
    def get_attention_maps(self, rgb_image: Tensor, visible_mask: Tensor, layer_idx: int = -1) -> Tensor:
        """
        Estrae le attention maps per visualizzazione
        """
        x = torch.cat([rgb_image, visible_mask], dim=1)
        x = self.encoder.patch_embedding(x)
        
        B = x.shape[0]
        cls_tokens = self.encoder.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.encoder.pos_embedding
        x = self.encoder.dropout(x)
        
        # Modifica il forward per estrarre attention
        for i, block in enumerate(self.encoder.blocks):
            if i == layer_idx or (layer_idx == -1 and i == len(self.encoder.blocks) - 1):
                # Hook per estrarre attention weights
                def hook_fn(module, input, output):
                    return output
                
                handle = block.attn.register_forward_hook(hook_fn)
                x = block(x)
                handle.remove()
                break
            else:
                x = block(x)
                
        return x
    
    def unfreeze_pretrained(self):
        """Scongela tutti i parametri per fine-tuning completo"""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("🔓 Unfrozen all pretrained parameters")
    
    def get_trainable_params(self) -> Dict[str, int]:
        """Ritorna info sui parametri addestrabili"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        
        return {
            'total': total,
            'trainable': trainable,
            'frozen': frozen,
            'trainable_pct': (trainable / total) * 100
        }

class AmodalSegmentationLoss(nn.Module):
    def __init__(self, weights: Dict[str, float] = None):
        super().__init__()
        # Pesi di default per ogni componente della loss
        self.weights = {
            'dice': 1.0,
            'bce': 1.0,
            'boundary': 0.5,
            'consistency': 0.5
        }
        if weights is not None:
            self.weights.update(weights)
        
    def forward(self, pred: Tensor, target: Tensor, visible_mask: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            pred: Predicted amodal mask [B, 1, H, W]
            target: Ground truth amodal mask [B, 1, H, W]
            visible_mask: Visible mask [B, 1, H, W]
        """
        losses = {}
        
        # Ensure all inputs are in valid range [0, 1]
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)
        visible_mask = torch.clamp(visible_mask, 0.0, 1.0)
        
        # Check for NaN/Inf values
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            print("WARNING: NaN/Inf detected in predictions during loss computation")
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
        
        if torch.isnan(target).any() or torch.isinf(target).any():
            print("WARNING: NaN/Inf detected in targets during loss computation")
            target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0)
        
        # 1. Dice Loss - per gestire lo sbilanciamento delle classi
        dice_loss = self.dice_loss(pred, target)
        losses['dice'] = dice_loss * self.weights['dice']
        
        # 2. BCE Loss - per pixel-wise accuracy (with eps for numerical stability)
        eps = 1e-7
        pred_stable = torch.clamp(pred, eps, 1.0 - eps)
        bce_loss = F.binary_cross_entropy(pred_stable, target)
        losses['bce'] = bce_loss * self.weights['bce']
        
        # 3. Boundary Loss - per bordi più precisi
        boundary_loss = self.boundary_loss(pred, target)
        losses['boundary'] = boundary_loss * self.weights['boundary']
        
        # 4. Consistency Loss - la parte visibile deve essere consistente
        consistency_loss = self.consistency_loss(pred, visible_mask)
        losses['consistency'] = consistency_loss * self.weights['consistency']
        
        # Loss totale
        losses['total'] = sum(losses.values())
        
        return losses
    
    @staticmethod
    def dice_loss(pred: Tensor, target: Tensor) -> Tensor:
        smooth = 1e-5
        intersection = (pred * target).sum(dim=(2,3))
        union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()
    
    @staticmethod
    def boundary_loss(pred: Tensor, target: Tensor) -> Tensor:
        # Calcola i gradienti per trovare i bordi
        sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).reshape(1,1,3,3)
        sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).reshape(1,1,3,3)
        
        if pred.is_cuda:
            sobel_x = sobel_x.cuda()
            sobel_y = sobel_y.cuda()
            
        pred_edges_x = F.conv2d(pred, sobel_x, padding=1)
        pred_edges_y = F.conv2d(pred, sobel_y, padding=1)
        target_edges_x = F.conv2d(target, sobel_x, padding=1)
        target_edges_y = F.conv2d(target, sobel_y, padding=1)
        
        # Use torch.clamp to prevent negative values before sqrt
        pred_edges_squared = torch.clamp(pred_edges_x**2 + pred_edges_y**2, min=0.0)
        target_edges_squared = torch.clamp(target_edges_x**2 + target_edges_y**2, min=0.0)
        
        # Add epsilon for numerical stability
        eps = 1e-8
        pred_edges = torch.sqrt(pred_edges_squared + eps)
        target_edges = torch.sqrt(target_edges_squared + eps)
        
        return F.mse_loss(pred_edges, target_edges)
    
    @staticmethod
    def consistency_loss(pred: Tensor, visible_mask: Tensor) -> Tensor:
        # La predizione deve includere la parte visibile
        # Add epsilon for numerical stability
        eps = 1e-7
        pred_stable = torch.clamp(pred * visible_mask, eps, 1.0 - eps)
        visible_mask_stable = torch.clamp(visible_mask, eps, 1.0 - eps)
        return F.binary_cross_entropy(pred_stable, visible_mask_stable)

# Funzione di utilità per creare il modello
def create_amodal_vit(config: str = 'base') -> AmodalViT:
    """
    Crea un modello AmodalViT con configurazioni predefinite
    
    Args:
        config: 'tiny', 'small', 'base', 'large'
    """
    configs = {
        'tiny': dict(embed_dim=192, num_layers=12, num_heads=3),
        'small': dict(embed_dim=384, num_layers=12, num_heads=6),
        'base': dict(embed_dim=768, num_layers=12, num_heads=12),
        'large': dict(embed_dim=1024, num_layers=24, num_heads=16),
    }
    
    if config not in configs:
        raise ValueError(f"Config {config} not supported. Choose from {list(configs.keys())}")
        
    return AmodalViT(**configs[config])


# Test del modello
if __name__ == "__main__":
    # Test con dati random
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = create_amodal_vit('base').to(device)
    
    # Input di test
    batch_size = 2
    rgb = torch.randn(batch_size, 3, 224, 224).to(device)
    visible_mask = torch.randn(batch_size, 1, 224, 224).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    with torch.no_grad():
        amodal_pred = model(rgb, visible_mask)
        print(f"Input RGB shape: {rgb.shape}")
        print(f"Input visible mask shape: {visible_mask.shape}")
        print(f"Output amodal mask shape: {amodal_pred.shape}")
        print(f"Output range: [{amodal_pred.min():.3f}, {amodal_pred.max():.3f}]")