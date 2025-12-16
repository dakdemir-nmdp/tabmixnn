import torch
from torch import nn
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from .formula import CompiledDesign
from .model import MixedEffectsRepresentation, SAMLayer, TabMixNNModule


@dataclass
class EncodedBatch:
    """
    Container returned by MixedEffectsEncoder.encode_dataframe.
    """
    X: torch.Tensor
    Z_indices: List[torch.Tensor]
    Z_slopes: List[torch.Tensor]
    targets: Dict[str, torch.Tensor]


def _build_random_config(design: CompiledDesign) -> List[Dict[str, Any]]:
    cfg = []
    for term in design.random_terms:
        cfg.append(
            {
                "term_id": term.term_id,
                "n_levels": max(term.level_map.values()) + 1 if term.level_map else 1,
                "n_slopes": len(term.slope_cols),
                "group": term.group_col,
                "train_levels": getattr(term, "train_levels", []),
            }
        )
    return cfg


def _infer_fixed_input_dim(design: CompiledDesign) -> int:
    dim = 1 if design.intercept_included else 0
    for spec in design.fixed_cols:
        if spec["type"] == "numeric":
            dim += 1
        else:
            # Use the actual dummy columns emitted by the design (drops base if intercept)
            dummy_levels = design.categorical_dummy_levels.get(
                spec["name"], design.categorical_levels[spec["name"]]
            )
            dim += len(dummy_levels)
    return dim


class MixedEffectsEncoder(nn.Module):
    """
    Reusable tabular encoder that wraps the mixed-effects block (and optional SAM)
    so it can be dropped into arbitrary PyTorch models as the first layer.

    Typical usage:
        encoder = MixedEffectsEncoder.from_formula("~ x1 + (1|g)", df_train)
        batch = encoder.encode_dataframe(df_batch)
        h, kl = encoder(batch.X, batch.Z_indices, batch.Z_slopes)
        out = downstream_mlp(h)
        loss = task_loss(out, y) + lambda_kl * kl + lambda_sparse * encoder.sparsity_loss()
    """

    def __init__(
        self,
        compiled_design: CompiledDesign,
        embedding_dim: int = 64,
        cov_struct_map: Optional[Dict[str, Any]] = None,
        add_sam: bool = False,
        fixed_input_dim: Optional[int] = None,
    ):
        super().__init__()
        self.design = compiled_design

        fixed_dim = fixed_input_dim or _infer_fixed_input_dim(compiled_design)
        random_config = _build_random_config(compiled_design)

        self.mixed_block = MixedEffectsRepresentation(
            fixed_input_dim=fixed_dim,
            embedding_dim=embedding_dim,
            random_terms_config=random_config,
            cov_struct_map=cov_struct_map,
            kl_normalize=True,  # Hardcoded optimal default
            scale_random_by_terms=False,  # Hardcoded optimal default
        )

        self.mixed_activation = None  # Hardcoded: no activation after mixed block

        self.sam_proj = None
        self.sam = None
        self.output_dim = embedding_dim
        if add_sam and embedding_dim > 1:
            # SAM always matches embedding_dim (no separate sam_dim parameter)
            self.sam = SAMLayer(embedding_dim)
            self.output_dim = embedding_dim

    def forward(self, X, Z_indices, Z_slopes):
        h, kl = self.mixed_block(X, Z_indices, Z_slopes)
        if self.mixed_activation is not None:
            h = self.mixed_activation(h)
        if self.sam is not None:
            h = self.sam(h)
        return h, kl

    def encode_dataframe(self, df, device: Optional[torch.device] = None, include_targets: bool = False) -> EncodedBatch:
        """
        Transform a DataFrame into tensors aligned with the encoder.
        """
        X, Z_ind, Z_slopes, Ys = self.design.transform(df, mode="predict")
        device = device or self._infer_device()

        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        Z_ind_t = [torch.tensor(z, dtype=torch.long, device=device) for z in Z_ind]
        Z_slopes_t = [torch.tensor(z, dtype=torch.float32, device=device) for z in Z_slopes]

        targets_t: Dict[str, torch.Tensor] = {}
        if include_targets:
            for k, v in Ys.items():
                targets_t[k] = torch.tensor(v, dtype=torch.float32, device=device)

        return EncodedBatch(X=X_t, Z_indices=Z_ind_t, Z_slopes=Z_slopes_t, targets=targets_t)

    def sparsity_loss(self, lambda_sparse: float = 1.0):
        if self.sam is None:
            return torch.tensor(0.0, device=self._infer_device())
        return lambda_sparse * self.sam.penalty()

    def _infer_device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @classmethod
    def from_formula(
        cls,
        formula: str,
        df,
        embedding_dim: int = 64,
        cov_struct: Optional[Dict[str, Any]] = None,
        add_sam: bool = False,
    ):
        design = CompiledDesign(
            formula=formula,
            outcomes=[],
            sam_location="first",
            sam_dim=None,  # Always match layer width
            cov_struct_map=cov_struct,
        )
        design.fit(df)
        return cls(
            compiled_design=design,
            embedding_dim=embedding_dim,
            cov_struct_map=cov_struct,
            add_sam=add_sam,
        )


__all__ = ["MixedEffectsEncoder", "EncodedBatch"]
