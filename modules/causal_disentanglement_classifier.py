"""
Causal Disentanglement Classifier

Backbone features are encoded into two representations by separate MLP encoders
("causal_features" and "non_causal_features"). Linear heads map each encoding
to logits: "causal_logits" and "non_causal_logits". Hidden blocks share the
same pattern: Linear -> LayerNorm -> ReLU -> Dropout, then a final Linear; no
dual-path or intervention module is used.

The two encoder branches use the same MLP skeleton; widths are controlled only
by "causal_mlp_layer_dims" and "non_causal_mlp_layer_dims". Each branch uses
the same built-in "SingleBranch" default only when that argument is
omitted. The encoders are symmetric only when both lists are omitted (or
identical); if one branch supplies a custom list and the other does not, their
widths can differ.

An encoder could in principle be implemented as an MLP, a Transformer, or
another architecture; here an MLP is used for a lightweight parameter and compute
footprint and for straightforward coupling to video backbones.

Training-time disentanglement (e.g. bias head and HSIC) is meant to be used
together with counterfactual videos (CFI): without paired cf_videos, both heads
see the same inputs and the same binary labels, so they do not have the
intended causal vs non-causal separation.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class SingleBranch(nn.Module):
    """
    One MLP encoder branch: backbone features -> a single representation.

    Each hidden block is Linear -> LayerNorm -> ReLU -> Dropout; the output
    layer is Linear only. mlp_dropout applies inside those hidden blocks.

    If mlp_layer_dims is None, layer widths default to five stages: embed_dim,
    then three hidden layers of width max(1, embed_dim // 2), then embed_dim again
    (input and output width both embed_dim). Otherwise mlp_layer_dims lists all
    widths end-to-end; the first width must equal embed_dim.
    """

    def __init__(
        self,
        embed_dim: int,
        mlp_layer_dims: Optional[List[int]] = None,
        mlp_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp_dropout = mlp_dropout

        if mlp_layer_dims is None:
            hidden = max(1, embed_dim // 2)
            mlp_layer_dims = [embed_dim] + [hidden] * 3 + [embed_dim]
        else:
            if mlp_layer_dims[0] != embed_dim:
                raise ValueError(
                    f"mlp_layer_dims[0] must equal embed_dim ({embed_dim}), "
                    f"got {mlp_layer_dims[0]}."
                )
            if len(mlp_layer_dims) < 2:
                raise ValueError(
                    f"mlp_layer_dims must have length >= 2, got {len(mlp_layer_dims)}."
                )

        self.mlp_layer_dims = list(mlp_layer_dims)

        layers: List[nn.Module] = []
        prev_dim = mlp_layer_dims[0]
        for hidden_dim in mlp_layer_dims[1:-1]:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(mlp_dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, mlp_layer_dims[-1]))
        self.mlp = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map features from [B, embed_dim] to [B, D] (D = final MLP width)."""
        return self.mlp(features)


class CausalDisentanglementClassifier(nn.Module):
    """
    Two SingleBranch MLP encoders (causal / non-causal) and two linear heads.
    Encoder widths come from causal_mlp_layer_dims / non_causal_mlp_layer_dims; when
    omitted, that branch uses the SingleBranch default (see SingleBranch). Encoders are
    symmetric only when both dimension lists are omitted (or identical); a custom list on
    one branch and None on the other yields asymmetric MLPs.

    Module attributes causal_branch and non_causal_branch are these encoders.

    Outputs:
        - "causal_features" [B, d_c], "non_causal_features" [B, d_nc].
        - "causal_logits" [B, num_classes], "non_causal_logits" [B, num_classes].

    causal_logits is always causal_cls_head(causal_features) in train and eval.

    causal_mlp_dropout and non_causal_mlp_dropout set mlp_dropout on the corresponding
    SingleBranch encoder (hidden blocks only).
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_classes: int = 1,
        causal_mlp_layer_dims: Optional[List[int]] = None,
        causal_mlp_dropout: float = 0.1,
        non_causal_mlp_layer_dims: Optional[List[int]] = None,
        non_causal_mlp_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.causal_mlp_dropout = causal_mlp_dropout
        self.non_causal_mlp_dropout = non_causal_mlp_dropout

        self.causal_branch = SingleBranch(
            embed_dim=embed_dim,
            mlp_layer_dims=causal_mlp_layer_dims,
            mlp_dropout=causal_mlp_dropout,
        )

        self.non_causal_branch = SingleBranch(
            embed_dim=embed_dim,
            mlp_layer_dims=non_causal_mlp_layer_dims,
            mlp_dropout=non_causal_mlp_dropout,
        )

        self.causal_cls_head = nn.Linear(
            self.causal_branch.mlp[-1].out_features, num_classes
        )
        self.non_causal_cls_head = nn.Linear(
            self.non_causal_branch.mlp[-1].out_features, num_classes
        )

    def forward(self, features: torch.Tensor) -> Dict[str, Any]:
        """
        Args:
            features (torch.Tensor): Backbone features, shape [B, embed_dim].

        Returns:
            Dict[str, torch.Tensor]: "causal_features", "non_causal_features",
            "causal_logits", "non_causal_logits" with shapes [B, d_c], [B, d_nc],
            [B, num_classes], [B, num_classes].
        """
        if features.shape[-1] != self.embed_dim:
            raise ValueError(
                f"CausalDisentanglementClassifier expected input last dim {self.embed_dim} "
                f"(embed_dim), got {features.shape[-1]}."
            )

        causal_features = self.causal_branch(features)
        non_causal_features = self.non_causal_branch(features)

        causal_logits = self.causal_cls_head(causal_features)
        non_causal_logits = self.non_causal_cls_head(non_causal_features)

        return {
            "causal_features": causal_features,
            "causal_logits": causal_logits,
            "non_causal_features": non_causal_features,
            "non_causal_logits": non_causal_logits,
        }
