# __init__.py - material classifier stage: exposes the hybrid model builder
"""Vision models for waste classification."""

from edgewaste.classification.hybrid_model import HybridConvNeXtViT, build_model

__all__ = ["HybridConvNeXtViT", "build_model"]
