"""
Evaluation for AI-generated video detection with optional CFI and t-SNE data collection.

Video batches use BTCHW layout [B, T, C, H, W] from the validation loader.

eval_model runs the detector in eval mode and aggregates validation metrics.
Main head: val_acc, val_real_acc, val_fake_acc, val_loss, val_auc, val_ap, val_f1
(some ranking metrics may be None when undefined). val_loss is BCE on "logits"
(linear) or "causal_logits" (CD). When CD, bias head returns val_bias_acc,
val_bias_real_acc, val_bias_fake_acc, val_bias_loss, val_bias_auc, val_bias_ap,
val_bias_f1; val_bias_loss mirrors training L_bias (not L_ind).
CD disentanglement is intended when training uses CFI so batches include cf_videos
(see run_functions/train.py and modules/detector.py).
When collect_data is True and the model is VideoDetector, features are
collected for t-SNE: linear head uses video_level_reps; CD head uses
causal_features and non_causal_features (both returned when CD). Baseline
models do not provide these; set collect_data=False for baselines.

Sample scope (where each metric is computed):
  - val_acc, val_real_acc, val_fake_acc: original batch only (main head:
    causal_logits when CD, else logits).
  - val_loss: BCE on the main head; when cfi_pipeline is provided, original +
    counterfactual videos; else original only.
  - val_auc, val_ap, val_f1: original batch only (main head).
  - val_bias_acc, val_bias_real_acc, val_bias_fake_acc: original batch only when CD
    (bias head); same None rules as val_real_acc / val_fake_acc on the main head.
  - val_bias_loss (CD only): BCE on non_causal_logits; same row scope as val_loss;
    CF rows labeled 0 (train L_bias rule).
  - val_bias_auc, val_bias_ap, val_bias_f1: original batch only when CD (bias head).
  - Overall val_auc, val_ap, val_f1 (and val_bias_auc, val_bias_ap, val_bias_f1 when CD):
    None when the concatenated original batch has fewer than 2 samples or only one
    class; F1 uses the same definability rule as AUC/AP.
  - val_cf_acc: counterfactual videos only (when cfi_pipeline is used).
  - val_bias_cf_acc: counterfactual rows only when CD and cfi_pipeline append CF videos;
    bias head; complement threshold vs val_cf_acc, aligned with L_bias CF labels.
  - val_by_gs, val_bias_by_gs: original batch only; one entry per distinct gs in
    gs_labels (including -1 for real). Each entry: gs_acc, gs_num_samples.
    Empty dict when gs_labels is empty (_by_gs_metrics returns early).

Optional cfi_pipeline: when provided, counterfactual videos are generated and evaluated;
val_cf_acc, val_bias_cf_acc (when CD), and optional counterfactual features (gs label 999) are returned.
Detector runs a single forward on [original batch, counterfactual videos] when CFI is used.
When the detector uses CD, val_bias_acc, val_bias_real_acc, val_bias_fake_acc,
val_bias_loss, val_bias_auc, val_bias_ap, val_bias_f1, val_bias_by_gs (when return_by_gs),
and val_bias_cf_acc (with CFI) are returned. CFI generation uses the same AMP context as the detector
(CUDA: bfloat16 autocast when torch.cuda.is_bf16_supported(); otherwise fp32). CFI debug
frame dumps are best-effort during training only (see run_functions/train.py); eval_model
does not write frames to disk.

det_model may be DataParallel-wrapped; use get_inner_model(det_model) for
attribute access (e.g. _classifier_type). cfi_pipeline is not DP-wrapped.
"""

from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from loss_functions import classification_loss
from my_utils import get_inner_model, metrics_accumulator


def _binary_classification_metrics(
    preds: np.ndarray, labels: np.ndarray
) -> Tuple[
    float, Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]
]:
    """Aggregate binary metrics from scores and ground-truth labels (original batch).

    Args:
        preds (np.ndarray): 1D sigmoid scores in [0, 1]; fake is predicted when score > 0.5.
        labels (np.ndarray): 1D binary labels (0 real, 1 fake), same length as preds.

    Returns:
        Tuple[
            float,
            Optional[float],
            Optional[float],
            Optional[float],
            Optional[float],
            Optional[float],
        ]:
        acc, real_acc, fake_acc, auc, ap, f1. auc, ap, and f1 are None when there are
        fewer than 2 samples or only one class in labels. real_acc/fake_acc are None when
        the corresponding class is absent in labels.
    """
    pred_labels = (preds > 0.5).astype(int)
    acc = float(accuracy_score(labels, pred_labels))
    real_mask = labels == 0
    fake_mask = labels == 1
    real_acc = (
        float((pred_labels[real_mask] == 0).mean()) if real_mask.any() else None
    )
    fake_acc = (
        float((pred_labels[fake_mask] == 1).mean()) if fake_mask.any() else None
    )
    num_classes = len(np.unique(labels))
    definable_ranking = len(labels) >= 2 and num_classes >= 2
    if definable_ranking:
        auc = float(roc_auc_score(labels, preds))
        ap = float(average_precision_score(labels, preds))
        f1 = float(f1_score(labels, pred_labels, zero_division=0.0))
    else:
        auc = None
        ap = None
        f1 = None
    return acc, real_acc, fake_acc, auc, ap, f1


def _by_gs_metrics(
    preds: np.ndarray, labels: np.ndarray, gs_labels: np.ndarray
) -> Dict[int, Dict[str, Any]]:
    """Per-generation-source binary metrics (aligned rows).

    One entry per distinct value in gs_labels (including -1 for real videos after
    convert_labeled_video_paths). For each gs, gs_acc is accuracy_score on that slice
    (threshold sigmoid > 0.5 vs base_label); gs_num_samples is len(gs_base_labels).

    Args:
        preds (np.ndarray): 1D sigmoid scores.
        labels (np.ndarray): 1D binary labels (0 real, 1 fake).
        gs_labels (np.ndarray): 1D generation source ids, same length as preds and labels.

    Returns:
        Dict[int, Dict[str, Any]]: Maps gs id to gs_acc and gs_num_samples only.
            Empty dict when gs_labels has length zero (guard before grouping).
    """
    if gs_labels.size == 0:
        return {}

    by_gs: Dict[int, Dict[str, Any]] = {}
    for gs in np.unique(gs_labels):
        mask = gs_labels == gs
        gs_base_labels = labels[mask]
        gs_preds = preds[mask]
        gs_pred_labels = (gs_preds > 0.5).astype(int)
        gs_acc = float(accuracy_score(gs_base_labels, gs_pred_labels))
        gs_num_samples = len(gs_base_labels)
        by_gs[int(gs)] = {
            "gs_acc": gs_acc,
            "gs_num_samples": gs_num_samples,
        }
    return by_gs


def eval_model(
    det_model: torch.nn.Module,
    val_dataloader: DataLoader,
    device: torch.device,
    return_by_gs: bool = False,
    collect_data: bool = False,
    cfi_pipeline: Optional[torch.nn.Module] = None,
    desc: str = "Validating",
) -> Dict[str, Any]:
    """
    Evaluate the detector on the validation set. val_loss is BCE on the main
    head only ("logits" or "causal_logits"). When CD, val_bias_loss mirrors
    training L_bias on non_causal_logits (not L_ind). Optionally evaluate on counterfactual
    videos and collect features for t-SNE. See module docstring for sample scope.

    t-SNE: linear uses video_level_reps; CD uses causal_features and
    non_causal_features (both returned when CD). Baseline models do not
    expose these; set collect_data=False for baselines.

    Args:
        det_model (torch.nn.Module): Detector (VideoDetector or Baseline).
        val_dataloader (DataLoader): Batch: video, base_label,
            generation_source_label; optional semantic_source_label.
        device (torch.device): Device.
        return_by_gs (bool): If True, return val_by_gs (and val_bias_by_gs when CD): per
            distinct gs in gs_labels (including -1 for real), keys gs_acc and gs_num_samples
            (see module docstring).
        collect_data (bool): If True, collect features and labels for t-SNE.
            Should be False for baseline models.
        cfi_pipeline (Optional[torch.nn.Module]): If provided and the batch has
            real videos, counterfactual videos are generated and evaluated; val_cf_acc,
            val_bias_cf_acc when CD, and optional counterfactual features (gs label 999) are returned.
        desc (str): Progress bar description.

    Returns:
        Dict[str, Any]: Main head: val_acc, val_real_acc, val_fake_acc, val_loss,
            val_auc, val_ap, val_f1. When CD: val_bias_acc, val_bias_real_acc,
            val_bias_fake_acc, val_bias_loss, val_bias_auc, val_bias_ap, val_bias_f1.
            If return_by_gs: val_by_gs (and val_bias_by_gs when CD). If cfi_pipeline:
            val_cf_acc; val_bias_cf_acc when CD. If collect_data: collected_features,
            collected_base_labels, collected_gs_labels; collected_non_causal_features
            and collected_bias_labels when CD.
            val_acc, val_real_acc, and val_fake_acc are over the original batch only; when CD,
            val_bias_acc, val_bias_real_acc, and val_bias_fake_acc use the same scope on the
            bias head. val_loss and val_bias_loss are mean BCE (see module docstring for row
            scope). Overall val_auc, val_ap, and val_f1 are None when the concatenated
            original batch has fewer than 2 samples or only one class (F1 uses the same
            definability rule as AUC/AP); val_bias_auc, val_bias_ap, and val_bias_f1 follow
            the same rule when CD.
            val_cf_acc and val_bias_cf_acc are over counterfactual rows only when present;
            val_by_gs and val_bias_by_gs (when return_by_gs) use the original batch only.
            When collect_data: features and base_labels over the full forward
            (original plus CF rows when CFI); gs_labels for original rows plus 999 for CF.
            CD also collects collected_non_causal_features and collected_bias_labels.
            See module docstring.
    """
    det_model.eval()
    if cfi_pipeline is not None:
        cfi_pipeline.eval()

    # Accumulator for detection metrics.
    metrics = metrics_accumulator()
    # Accumulator for t-SNE collection.
    if collect_data:
        collected = metrics_accumulator()

    use_bf16_amp = (
        device.type == "cuda"
        and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    )
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16_amp
        else nullcontext()
    )
    classifier_type = getattr(get_inner_model(det_model), "_classifier_type", None)

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc=desc, ncols=80):
            videos = batch["video"].contiguous().to(device)
            base_labels = torch.tensor(
                [int(base_label) for base_label in batch["base_label"]],
                dtype=torch.long,
                device=device,
            )
            real_mask = base_labels == 0
            batch_size = videos.size(0)
            gs_labels = torch.tensor(
                [int(gs_label) for gs_label in batch["generation_source_label"]],
                dtype=torch.long,
                device=device,
            )

            # Optionally generate counterfactual videos (inside AMP context) and run
            # a single detector forward on [original batch, cf_videos].
            cf_videos = None
            with amp_ctx:
                if cfi_pipeline is not None and real_mask.any():
                    real_videos = videos[real_mask]
                    real_indices = torch.where(real_mask)[0].cpu().tolist()
                    captions = (
                        [batch["semantic_source_label"][i] for i in real_indices]
                        if "semantic_source_label" in batch
                        else [""] * len(real_indices)
                    )
                    cf_videos = cfi_pipeline(real_videos, captions=captions)[
                        "cf_videos"
                    ]

                if cf_videos is not None:
                    num_cf_videos = cf_videos.size(0)
                    videos = torch.cat([videos, cf_videos], dim=0)
                    base_labels = torch.cat(
                        [
                            base_labels,
                            torch.ones(
                                num_cf_videos,
                                dtype=torch.long,
                                device=device,
                            ),
                        ],
                        dim=0,
                    )
                else:
                    num_cf_videos = 0

                outputs = det_model(videos)
                main_logits = (
                    outputs["causal_logits"]
                    if classifier_type == "cd"
                    else outputs["logits"]
                )
                val_loss = classification_loss(main_logits, base_labels)
                metrics.add("val_loss", val_loss.item())

                metrics.add("ori_base_labels", base_labels[:batch_size].cpu())
                metrics.add(
                    "ori_preds",
                    torch.sigmoid(main_logits[:batch_size, 0]).cpu(),
                )
                if num_cf_videos > 0:
                    metrics.add(
                        "cf_preds",
                        torch.sigmoid(main_logits[batch_size:, 0]).cpu(),
                    )
                if classifier_type == "cd":
                    bias_labels = base_labels.clone()
                    if num_cf_videos > 0:
                        bias_labels[batch_size:] = 0
                    val_bias_loss = classification_loss(
                        outputs["non_causal_logits"], bias_labels
                    )
                    metrics.add("val_bias_loss", val_bias_loss.item())

                    metrics.add(
                        "bias_preds",
                        torch.sigmoid(
                            outputs["non_causal_logits"][:batch_size, 0]
                        ).cpu(),
                    )
                    if num_cf_videos > 0:
                        metrics.add(
                            "bias_cf_preds",
                            torch.sigmoid(
                                outputs["non_causal_logits"][batch_size:, 0]
                            ).cpu(),
                        )

                if return_by_gs:
                    metrics.add("gs_labels", gs_labels.cpu().float())

                if collect_data:
                    if classifier_type == "cd":
                        collected.add(
                            "collected_features",
                            outputs["causal_features"].cpu(),
                        )
                        collected.add(
                            "collected_non_causal_features",
                            outputs["non_causal_features"].cpu(),
                        )
                        collected.add("collected_bias_labels", bias_labels.cpu())
                    else:
                        collected.add(
                            "collected_features",
                            outputs["video_level_reps"].cpu(),
                        )
                    collected.add("collected_base_labels", base_labels.cpu())
                    collected.add("collected_gs_labels", gs_labels.cpu())
                    if num_cf_videos > 0:
                        cf_gs_labels = torch.full(
                            (num_cf_videos,), 999, dtype=torch.long, device="cpu"
                        )
                        collected.add("collected_gs_labels", cf_gs_labels)

    ori_true_labels = metrics.get_concat_tensor("ori_base_labels").float().numpy()
    ori_preds = metrics.get_concat_tensor("ori_preds").float().numpy()
    val_acc, val_real_acc, val_fake_acc, val_auc, val_ap, val_f1 = (
        _binary_classification_metrics(ori_preds, ori_true_labels)
    )

    val_results: Dict[str, Any] = {
        "val_acc": val_acc,
        "val_real_acc": val_real_acc,
        "val_fake_acc": val_fake_acc,
        "val_loss": metrics.get_mean("val_loss"),
        "val_auc": val_auc,
        "val_ap": val_ap,
        "val_f1": val_f1,
    }

    if metrics.exists("cf_preds"):
        cf_preds = metrics.get_concat_tensor("cf_preds").float().numpy()
        val_results["val_cf_acc"] = float((cf_preds > 0.5).mean())

    if metrics.exists("bias_preds"):
        bias_preds = metrics.get_concat_tensor("bias_preds").float().numpy()
        (
            val_bias_acc,
            val_bias_real_acc,
            val_bias_fake_acc,
            val_bias_auc,
            val_bias_ap,
            val_bias_f1,
        ) = _binary_classification_metrics(bias_preds, ori_true_labels)
        val_results["val_bias_acc"] = val_bias_acc
        val_results["val_bias_real_acc"] = val_bias_real_acc
        val_results["val_bias_fake_acc"] = val_bias_fake_acc
        val_results["val_bias_loss"] = metrics.get_mean("val_bias_loss")
        val_results["val_bias_auc"] = val_bias_auc
        val_results["val_bias_ap"] = val_bias_ap
        val_results["val_bias_f1"] = val_bias_f1
    if metrics.exists("bias_cf_preds"):
        bias_cf_preds = metrics.get_concat_tensor("bias_cf_preds").float().numpy()
        val_results["val_bias_cf_acc"] = float(1.0 - (bias_cf_preds > 0.5).mean())

    if return_by_gs and metrics.exists("gs_labels"):
        epoch_gs_labels = metrics.get_concat_tensor("gs_labels").float().numpy()
        val_results["val_by_gs"] = _by_gs_metrics(
            ori_preds, ori_true_labels, epoch_gs_labels
        )
        if classifier_type == "cd" and metrics.exists("bias_preds"):
            val_results["val_bias_by_gs"] = _by_gs_metrics(
                bias_preds, ori_true_labels, epoch_gs_labels
            )

    if collect_data and collected.exists("collected_features"):
        val_results["collected_features"] = collected.get_concat_tensor(
            "collected_features"
        )
        val_results["collected_base_labels"] = collected.get_concat_tensor(
            "collected_base_labels"
        )
        val_results["collected_gs_labels"] = collected.get_concat_tensor(
            "collected_gs_labels"
        )
        if collected.exists("collected_non_causal_features"):
            val_results["collected_non_causal_features"] = collected.get_concat_tensor(
                "collected_non_causal_features"
            )
            val_results["collected_bias_labels"] = collected.get_concat_tensor(
                "collected_bias_labels"
            )

    return val_results
