"""
Detector Module: Backbone + Pluggable Classifier

Classifier from classifier_type (and classifier_init_params).
- CD off: classifier_type="linear". Single nn.Linear(embed_dim, num_classes); no config needed.
- CD on: classifier_type="cd". CausalDisentanglementClassifier built with embed_dim and
  num_classes injected from backbone/detector. For the intended disentanglement
  behavior, training should include CFI counterfactuals (cf_videos); otherwise
  both encoder heads face the same classification task on identical inputs.
  Omit classifier_init_params (or null) for CD defaults; a non-empty dict may
  supply only CD-specific params (e.g. causal_mlp_layer_dims,
  non_causal_mlp_layer_dims, causal_mlp_dropout, non_causal_mlp_dropout); keys must
  match the CausalDisentanglementClassifier constructor.
  replace_classifier("cd", ...). In CD stage both backbone and CD classifier are trainable.
  Checkpoint stores a copy of passed-in classifier_init_params (embed_dim/num_classes re-injected on load).

Train/test: VideoDetector(**ckpt["init_params"]) then load_state_dict.
"""
from typing import Any, Dict, Optional, Type

import torch
import torch.nn as nn

from modules.causal_disentanglement_classifier import CausalDisentanglementClassifier
from video_backbones import (
    XCLIPVideoBackbone,
    CLIPVideoBackbone,
    DeMambaXCLIPVideoBackbone,
    DeMambaCLIPVideoBackbone,
)


BACKBONE_MAPPING: Dict[str, Type[nn.Module]] = {
    "xclip": XCLIPVideoBackbone,
    "clip": CLIPVideoBackbone,
    "demamba_xclip": DeMambaXCLIPVideoBackbone,
    "demamba_clip": DeMambaCLIPVideoBackbone,
}


class VideoDetector(nn.Module):
    """
    Backbone + classifier from classifier_type (and optional classifier_init_params).
    - CD off (classifier_type="linear"): Single nn.Linear(embed_dim, num_classes);
      embed_dim from backbone, num_classes from detector; classifier_init_params
      may be null in YAML for consistency (ignored at runtime).
    - CD on (classifier_type="cd"): CausalDisentanglementClassifier with embed_dim and
      num_classes always injected from backbone/detector. Intended use pairs CD with
      CFI during training so cf_videos provide a counterfactual contrast; without
      it, causal and non-causal heads are not meaningfully separated by task.
      classifier_init_params may contain only CD-specific params (e.g. causal_mlp_layer_dims,
      non_causal_mlp_layer_dims, causal_mlp_dropout, non_causal_mlp_dropout).
      Symmetric with linear: both classifiers receive embed_dim and num_classes from the detector.

    Forward keys: CD off -> "logits", "video_level_reps", "frame_level_reps",
    "patch_level_reps"; CD on -> "video_level_reps", "frame_level_reps",
    "patch_level_reps" plus CD outputs "causal_features", "causal_logits",
    "non_causal_features", "non_causal_logits" (no "logits" key when CD on).

    Args:
        video_backbone (str): "xclip", "clip", "demamba_xclip", "demamba_clip".
        hf_repo (Optional[str]): Pretrained path. None => backbone default.
        backbone_ft (str): For XCLIP: "frozen", "lora", "full", or "mit_only".
            For CLIP: "frozen" or "full".
            For DeMamba-XCLIP / DeMamba-CLIP: "frozen", "full", or "mamba_only".
        num_classes (int): Output classes for main task (default 1).
        classifier_type (str): "linear" or "cd". Determines which classifier is built
            internally. Same key as in init_params for unified train/test instantiation.
        classifier_init_params (Optional[Dict[str, Any]]): Omit or set null for
            CD default hyperparameters (embed_dim and num_classes are still
            injected). For "linear", ignored. For "cd", a non-empty dict supplies
            CD-only kwargs (e.g. causal_mlp_layer_dims, non_causal_mlp_layer_dims,
            causal_mlp_dropout, non_causal_mlp_dropout). Keys must match the CD
            classifier constructor. Prefer null over an empty mapping when using
            defaults.
    """

    def __init__(
        self,
        video_backbone: str = "xclip",
        hf_repo: Optional[str] = None,
        backbone_ft: str = "frozen",
        num_classes: int = 1,
        classifier_type: str = "linear",
        classifier_init_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if video_backbone not in BACKBONE_MAPPING:
            raise ValueError(
                f"Unsupported video_backbone: {video_backbone}. "
                f"Choose from: {list(BACKBONE_MAPPING.keys())}"
            )
        backbone_class = BACKBONE_MAPPING[video_backbone]
        if hf_repo is not None:
            self.backbone = backbone_class(hf_repo=hf_repo, finetune_mode=backbone_ft)
        else:
            self.backbone = backbone_class(finetune_mode=backbone_ft)
        embed_dim = self.backbone.embed_dim

        self._video_backbone = video_backbone
        self._hf_repo = hf_repo
        self._backbone_ft = backbone_ft
        self._num_classes = num_classes
        self._classifier_type = classifier_type
        self._classifier_init_params: Optional[Dict[str, Any]] = None

        if classifier_type == "linear":
            self.classifier = nn.Linear(embed_dim, num_classes)
        elif classifier_type == "cd":
            # Inject embed_dim and num_classes from detector/backbone; store a copy
            # of the passed-in params for checkpoint (embed_dim/num_classes re-injected on load).
            params = dict(classifier_init_params) if classifier_init_params else {}
            params["embed_dim"] = embed_dim
            params["num_classes"] = num_classes
            self.classifier = CausalDisentanglementClassifier(**params)
            self._classifier_init_params = (
                classifier_init_params.copy() if classifier_init_params else None
            )
        else:
            raise ValueError(
                f"classifier_type must be 'linear' or 'cd', got {classifier_type!r}."
            )

        self._init_params = self._build_init_params()

    def _build_init_params(self) -> Dict[str, Any]:
        """Build the serializable init params dict (used in __init__ and replace_classifier)."""
        return {
            "video_backbone": self._video_backbone,
            "hf_repo": self._hf_repo,
            "backbone_ft": self._backbone_ft,
            "num_classes": self._num_classes,
            "classifier_type": self._classifier_type,
            "classifier_init_params": self._classifier_init_params,
        }

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Backbone -> classifier; merge classifier outputs with backbone features.

        Classifier outputs are forwarded unchanged. "video_level_reps",
        "frame_level_reps", and "patch_level_reps" are always added from the
        backbone. The classifier input is backbone "video_level_reps".

        Linear mode: classifier is nn.Linear; output includes "logits" plus
        backbone "video_level_reps", "frame_level_reps", and "patch_level_reps".

        CD mode: classifier returns "causal_features", "causal_logits",
        "non_causal_features", "non_causal_logits" (no top-level "logits" key), plus
        merged backbone "video_level_reps", "frame_level_reps", and
        "patch_level_reps".

        Args:
            pixel_values (torch.Tensor): Video batch, shape [B, T, C, H, W] (BTCHW).
                T and H, W must match the chosen video backbone.

        Returns:
            Dict[str, torch.Tensor]: Linear mode: "logits", "video_level_reps",
            "frame_level_reps", "patch_level_reps". CD mode:
            "causal_features", "causal_logits", "non_causal_features",
            "non_causal_logits", "video_level_reps", "frame_level_reps",
            "patch_level_reps".
        """
        backbone_outputs = self.backbone(pixel_values)
        features = backbone_outputs["video_level_reps"]
        if isinstance(self.classifier, nn.Linear):
            classifier_outputs = {"logits": self.classifier(features)}
        else:
            classifier_outputs = self.classifier(features)
        outputs = dict(classifier_outputs)
        outputs["video_level_reps"] = features
        outputs["frame_level_reps"] = backbone_outputs["frame_level_reps"]
        outputs["patch_level_reps"] = backbone_outputs["patch_level_reps"]
        return outputs

    def replace_classifier(
        self,
        classifier_type: str,
        classifier_init_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Replace the current classifier (e.g. enable CD: swap to CD classifier).

        Builds the new classifier from classifier_type and classifier_init_params.
        For "cd", embed_dim and num_classes are always taken from backbone and
        detector; classifier_init_params may contain only CD-specific kwargs for
        "CausalDisentanglementClassifier".
        Init params dict is updated so train and test use the same instantiation
        logic. The new classifier is moved to the same device as the rest of
        the model (e.g. after replace during training, backbone is already on CUDA).
        """
        if classifier_type == "linear":
            embed_dim = self.backbone.embed_dim
            self.classifier = nn.Linear(embed_dim, self._num_classes)
            self._classifier_type = "linear"
            self._classifier_init_params = None
        elif classifier_type == "cd":
            # Inject embed_dim and num_classes; store copy of passed-in params for checkpoint.
            params = dict(classifier_init_params) if classifier_init_params else {}
            params["embed_dim"] = self.backbone.embed_dim
            params["num_classes"] = self._num_classes
            self.classifier = CausalDisentanglementClassifier(**params)
            self._classifier_init_params = (
                classifier_init_params.copy() if classifier_init_params else None
            )
            self._classifier_type = "cd"
        else:
            raise ValueError(
                f"classifier_type must be 'linear' or 'cd', got {classifier_type!r}."
            )

        device = next(self.parameters()).device
        self.classifier = self.classifier.to(device)
        self._init_params = self._build_init_params()

    def get_init_params(self) -> Dict[str, Any]:
        """
        Return serializable init params for checkpoint.

        Used when saving checkpoint so that at test time the model can be
        instantiated with VideoDetector(**init_params) then load_state_dict.
        For "cd", classifier_init_params is a copy of the passed-in dict (typically
        CD-specific only); embed_dim and num_classes are injected at build time.
        """
        return self._init_params.copy()
