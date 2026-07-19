"""ConvNeXt + Vision Transformer hybrid with attention-based fusion.

Blueprint Module 3: "ConvNeXt + Vision Transformer + Attention -> feature
vector", then a classifier head over the fused vector. Two pretrained backbones
each produce a global feature vector; each is projected to a common dimension;
an attention module learns per-backbone weights so the model can lean on
whichever backbone is more informative for a given image; the attended,
concatenated vector feeds the classifier head.

Both backbones come from `timm` with `num_classes=0, global_pool="avg"`, so
each returns a pooled feature vector (not logits). This keeps the fusion logic
backbone-agnostic — swap `convnext_tiny` for `convnext_small` or the ViT for a
larger variant via config without touching this file.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm

from ..taxonomy import NUM_CLASSES


class AttentionFusion(nn.Module):
    """Learns scalar attention weights over N projected backbone features.

    Each backbone feature (already projected to `dim`) gets a learned attention
    score; scores are softmax-normalized across backbones and used to weight the
    features before concatenation. This is the "attention layer combining both"
    called for in docs/stage-1 Module 3.
    """

    def __init__(self, dim: int, num_streams: int = 2):
        super().__init__()
        self.num_streams = num_streams
        self.score = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        # feats: list of [B, dim]; stack to [B, N, dim]
        stacked = torch.stack(feats, dim=1)
        scores = self.score(stacked)  # [B, N, 1]
        weights = torch.softmax(scores, dim=1)  # attention over streams
        weighted = stacked * weights  # [B, N, dim]
        # concatenate weighted streams -> [B, N*dim]
        return weighted.flatten(start_dim=1), weights.squeeze(-1)


class HybridConvNeXtViT(nn.Module):
    def __init__(
        self,
        convnext: str = "convnext_tiny",
        vit: str = "vit_small_patch16_224",
        num_classes: int = NUM_CLASSES,
        fusion_dim: int = 512,
        dropout: float = 0.2,
        pretrained: bool = True,
        image_size: int = 224,
    ):
        super().__init__()
        self.convnext = timm.create_model(
            convnext, pretrained=pretrained, num_classes=0, global_pool="avg")
        self.vit = timm.create_model(
            vit, pretrained=pretrained, num_classes=0, global_pool="avg",
            img_size=image_size)

        cnx_dim = self.convnext.num_features
        vit_dim = self.vit.num_features

        # Project each backbone to a common dimension for fusion.
        self.proj_cnx = nn.Sequential(
            nn.Linear(cnx_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU())
        self.proj_vit = nn.Sequential(
            nn.Linear(vit_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU())

        self.fusion = AttentionFusion(fusion_dim, num_streams=2)

        # Classifier head over the fused (2 * fusion_dim) vector.
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        f_cnx = self.proj_cnx(self.convnext(x))
        f_vit = self.proj_vit(self.vit(x))
        fused, attn = self.fusion([f_cnx, f_vit])
        logits = self.head(fused)
        if return_attn:
            return logits, attn
        return logits

    @torch.no_grad()
    def feature_vector(self, x: torch.Tensor) -> torch.Tensor:
        """Fused feature vector — the hand-off point for Stage 2 sensor fusion."""
        f_cnx = self.proj_cnx(self.convnext(x))
        f_vit = self.proj_vit(self.vit(x))
        fused, _ = self.fusion([f_cnx, f_vit])
        return fused


def build_model(model_cfg, num_classes: int = NUM_CLASSES) -> HybridConvNeXtViT:
    """Construct the model from a ModelConfig."""
    return HybridConvNeXtViT(
        convnext=model_cfg.convnext,
        vit=model_cfg.vit,
        num_classes=num_classes,
        fusion_dim=model_cfg.fusion_dim,
        dropout=model_cfg.dropout,
        pretrained=model_cfg.pretrained,
        image_size=model_cfg.image_size,
    )
