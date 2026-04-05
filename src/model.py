"""
Vision Transformer (ViT) for Brain Tumor Classification and Segmentation
Multi-Modal MRI Support: T1, T2, FLAIR, T1-CE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    """
    Split image into patches and embed them.
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Linear projection of flattened patches
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e h w -> b (h w) e')
        )
    
    def forward(self, x):
        # x: (batch_size, channels, height, width)
        return self.projection(x)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    """
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attention_dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.projection_dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (batch_size, num_patches + 1, embed_dim)
        batch_size, num_tokens, embed_dim = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, num_tokens, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention: (Q * K^T) / sqrt(d_k)
        attention_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)
        
        # Apply attention to values
        attended_values = attention_probs @ v
        attended_values = attended_values.transpose(1, 2).reshape(batch_size, num_tokens, embed_dim)
        
        # Final projection
        output = self.projection(attended_values)
        output = self.projection_dropout(output)
        
        return output


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (Feed-forward network).
    """
    def __init__(self, embed_dim=768, hidden_dim=3072, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Transformer encoder block: Self-attention + MLP with residual connections.
    """
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)
    
    def forward(self, x):
        # Self-attention with residual connection
        x = x + self.attention(self.norm1(x))
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer for Brain Tumor Classification.
    Supports multi-modal MRI input (T1, T2, FLAIR, T1-CE).
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=4,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        dropout=0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)
        
        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights_module)
    
    def _init_weights_module(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # Transformer encoder blocks
        for block in self.blocks:
            x = block(x)
        
        # Layer normalization
        x = self.norm(x)
        
        # Classification head (use class token)
        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)
        
        return logits


class ViTSegmentation(nn.Module):
    """
    Vision Transformer for Brain Tumor Segmentation.
    Decoder for pixel-wise prediction.
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=4,  # Multi-modal: T1, T2, FLAIR, T1-CE
        num_classes=4,  # Background, Whole Tumor, Tumor Core, Enhancing Tumor
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        dropout=0.1
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Positional embedding (no class token for segmentation)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)
        
        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Decoder (upsampling to original resolution)
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(64, num_classes, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights_module)
    
    def _init_weights_module(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # Transformer encoder blocks
        for block in self.blocks:
            x = block(x)
        
        # Layer normalization
        x = self.norm(x)
        
        # Reshape for decoder
        num_patches_per_side = self.img_size // self.patch_size
        x = rearrange(x, 'b (h w) e -> b e h w', h=num_patches_per_side, w=num_patches_per_side)
        
        # Decoder
        segmentation_map = self.decoder(x)
        
        return segmentation_map


def count_parameters(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ImprovedViTSegmentation(nn.Module):
    """
    Improved Vision Transformer for Brain Tumor Segmentation with Skip Connections.
    U-Net style skip connections preserve spatial details from encoder.
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=4,
        num_classes=4,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        dropout=0.1
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)
        
        # Transformer encoder blocks with intermediate feature extraction
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Extract features at different depths for skip connections
        self.skip_indices = [2, 5, 8, 11]  # Extract at layers 3, 6, 9, 12
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Decoder with skip connections (U-Net style)
        # Stage 1: 14x14 -> 28x28
        self.decoder1 = nn.Sequential(
            nn.Conv2d(embed_dim, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Stage 2: 28x28 -> 56x56 (with skip from layer 9)
        self.skip_proj2 = nn.Conv2d(embed_dim, 256, 1)  # Project skip connection
        self.decoder2 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, 3, padding=1),  # Concatenate skip
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Stage 3: 56x56 -> 112x112 (with skip from layer 6)
        self.skip_proj3 = nn.Conv2d(embed_dim, 128, 1)
        self.decoder3 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Stage 4: 112x112 -> 224x224 (with skip from layer 3)
        self.skip_proj4 = nn.Conv2d(embed_dim, 64, 1)
        self.decoder4 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Final segmentation head
        self.seg_head = nn.Conv2d(64, num_classes, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights_module)
    
    def _init_weights_module(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        batch_size = x.shape[0]
        num_patches_per_side = self.img_size // self.patch_size
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # Transformer encoder blocks with skip connection extraction
        skip_features = []
        for idx, block in enumerate(self.blocks):
            x = block(x)
            # Extract features at specified depths
            if idx in self.skip_indices:
                # Reshape to spatial dimensions for skip connection
                skip_feat = rearrange(x, 'b (h w) e -> b e h w', 
                                    h=num_patches_per_side, w=num_patches_per_side)
                skip_features.append(skip_feat)
        
        # Layer normalization
        x = self.norm(x)
        
        # Reshape for decoder
        x = rearrange(x, 'b (h w) e -> b e h w', h=num_patches_per_side, w=num_patches_per_side)
        
        # Decoder Stage 1: 14x14 -> 28x28
        x = self.decoder1(x)
        x = self.upsample1(x)
        
        # Decoder Stage 2: 28x28 -> 56x56 (with skip from layer 9)
        skip2 = self.skip_proj2(skip_features[2])  # From layer 9
        skip2 = F.interpolate(skip2, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip2], dim=1)
        x = self.decoder2(x)
        x = self.upsample2(x)
        
        # Decoder Stage 3: 56x56 -> 112x112 (with skip from layer 6)
        skip3 = self.skip_proj3(skip_features[1])  # From layer 6
        skip3 = F.interpolate(skip3, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip3], dim=1)
        x = self.decoder3(x)
        x = self.upsample3(x)
        
        # Decoder Stage 4: 112x112 -> 224x224 (with skip from layer 3)
        skip4 = self.skip_proj4(skip_features[0])  # From layer 3
        skip4 = F.interpolate(skip4, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip4], dim=1)
        x = self.decoder4(x)
        x = self.upsample4(x)
        
        # Final segmentation
        segmentation_map = self.seg_head(x)
        
        return segmentation_map


def count_parameters(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_vit_classifier(num_classes=4, img_size=224, in_channels=3):
    """
    Factory function to create ViT classifier.
    """
    model = VisionTransformer(
        img_size=img_size,
        patch_size=16,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        dropout=0.1
    )
    return model


def create_vit_segmentation(num_classes=4, img_size=224, in_channels=4, use_skip_connections=True):
    """
    Factory function to create ViT segmentation model.
    
    Args:
        num_classes: Number of segmentation classes
        img_size: Input image size
        in_channels: Number of input channels (4 for multi-modal MRI)
        use_skip_connections: If True, use ImprovedViTSegmentation with U-Net style skip connections
    """
    if use_skip_connections:
        model = ImprovedViTSegmentation(
            img_size=img_size,
            patch_size=16,
            in_channels=in_channels,
            num_classes=num_classes,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            dropout=0.1
        )
    else:
        model = ViTSegmentation(
            img_size=img_size,
            patch_size=16,
            in_channels=in_channels,
            num_classes=num_classes,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            dropout=0.1
        )
    return model
