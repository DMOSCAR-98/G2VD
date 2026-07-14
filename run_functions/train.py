"""
Train one epoch for AI-generated video detection with optional CFI and CD.

Video batches from the dataloader use BTCHW layout [B, T, C, H, W] unless a path notes otherwise.

Only the detector is trained. CFIPipeline is inference-only and used to generate
counterfactual videos from real samples. Detector runs a single forward on the
concatenated batch [original, cf_videos] when CFI is enabled.

When enable_cd, optimization uses a three-term CD objective:
  L_total = w_cls*L_cls + w_bias*L_bias + w_ind*L_ind.
  Weights [w_cls, w_bias, w_ind] come from loss_weight_list when provided; if
  omitted, defaults to [1.0, 1.0, 1.0].

- L_cls: main classification loss (BCE on main_logits: causal_logits when CD,
  else logits), same batch as base_labels.
- L_bias: bias-head classification loss (non_causal_logits); labels match
  base_labels with counterfactual rows forced to 0 (real) when CFI appends
  cf_videos.
- L_ind: HSIC independence between causal_features and non_causal_features.

CD is intended to be used with CFI enabled (cf_videos appended to the batch).
Without counterfactual videos, both branches see the same per-sample binary
classification target and lack the real-vs-counterfactual contrast that
separates main-branch from bias-branch factors as intended, so disentanglement
is not meaningful in that sense.

Sample scope:
  - train_acc, bias_acc: original batch only.
  - cf_acc: counterfactual samples only (main head: fraction with sigmoid > 0.5).
  - bias_cf_acc: counterfactual samples only when CD and CFI; bias head,
    complement of the same threshold rule as cf_acc (fraction with sigmoid <= 0.5),
    aligned with bias_labels forcing CF rows to real (0) in L_bias.
  - losses: computed on the current training batch
    (original only, or original + counterfactual when CFI is enabled).

Gradient accumulation (grad_accum_steps > 1): each backward uses loss scaled by
1/grad_accum_steps. When the epoch is not a multiple of grad_accum_steps, the
trailing micro-batches are still applied with optimizer.step() and cleared at
epoch end so gradients never leak into the next epoch.

CFI counterfactual generation runs under the same AMP context as the detector
forward (CUDA: bfloat16 autocast when torch.cuda.is_bf16_supported(); otherwise
fp32 via nullcontext). CPU always uses fp32. The CFI forward itself is wrapped in
torch.no_grad() so no graph is built through the pipeline.

When enable_cfi, batch_idx == 0 attempts best-effort debug frame dumps under
./cfi_pipeline_outputs/{real_videos,rec_videos,far_videos,cf_videos} (one clip
each) for quick visual inspection; failures are logged as warnings and do not
abort training. Parallel jobs may overwrite the same directory. train_epoch runs
one epoch per call so this is the first batch of that epoch. Test-time eval
often omits cfi_pipeline, so debug dumps stay on the training path.
"""

import logging
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from loss_functions import classification_loss, independence_loss
from my_utils import metrics_accumulator, save_video_tensor_to_frames


def train_epoch(
    det_model: torch.nn.Module,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_accum_steps: int = 1,
    enable_cfi: bool = False,
    enable_cd: bool = False,
    cfi_pipeline: Optional[torch.nn.Module] = None,
    loss_weight_list: Optional[List[float]] = None,
    cf_cls_weight: float = 1.0,
    desc: str = "Training",
) -> Dict[str, Any]:
    """
    One epoch of training. Only the detector (det_model) is trained; CFI
    pipeline is eval-only. When enable_cfi: real videos are augmented with
    CFIPipeline counterfactual outputs (cf_videos), merged with the original
    batch for a single detector forward. L_cls is BCE on main_logits (causal_logits
    when enable_cd, else logits). When enable_cd: optimize with L_cls, L_bias,
    L_ind and weights [w_cls, w_bias, w_ind].

    Args:
        det_model (torch.nn.Module): Detector; compatible with flags (e.g.
            VideoDetector with backbone+CD when enable_cd=True;
            BaselineModelWrapper with enable_cd=False).
        train_dataloader (DataLoader): Training data loader (batch["video"],
            batch["base_label"], optional semantic_source_label for CFI captions).
        optimizer (torch.optim.Optimizer): Optimizer for det_model.
        device (torch.device): Device.
        grad_accum_steps (int): Gradient accumulation steps. Default 1. When the
            number of batches in the epoch is not divisible by grad_accum_steps,
            train_epoch still steps and zero_grad once after the loop for the
            final partial group (effective batch size for that step differs from
            a full accumulation window).
        enable_cfi (bool): Whether to use CFI counterfactual videos. Default False.
        enable_cd (bool): Whether to use Causal Disentanglement losses. Default False.
        cfi_pipeline (Optional[torch.nn.Module]): CFIPipeline when enable_cfi; None.
        loss_weight_list (Optional[List[float]]): When enable_cd: required length 3
            [w_cls, w_bias, w_ind] (typically from config). When omitted and
            enable_cd: defaults to [1.0, 1.0, 1.0]. When enable_cd=False: optional;
            default [1.0].
        cf_cls_weight (float): Weight for counterfactual rows in the main
            classification loss. Original rows always use weight 1.0. Default
            1.0 preserves the original unweighted behavior.
        desc (str): Progress bar description.

    Returns:
        Dict[str, Any]: train_acc, train_loss_components (dict: total_loss,
            cls_loss; when enable_cd: bias_loss, ind_loss);
            when enable_cfi: cf_acc; when enable_cd: bias_acc;
            when enable_cd and enable_cfi (with CF rows): bias_cf_acc.
            cf_acc is over counterfactual videos on the main head only; train_acc
            and bias_acc are over the original batch only; bias_cf_acc is over
            counterfactual rows on the bias head only (see module docstring). When
            enable_cfi, batch_idx == 0 attempts best-effort debug frames under
            ./cfi_pipeline_outputs/ (see module docstring). See module docstring for
            full sample scope.
    """
    if loss_weight_list is None:
        loss_weight_list = [1.0, 1.0, 1.0] if enable_cd else [1.0]
    cf_cls_weight = float(cf_cls_weight)

    if enable_cd:
        if len(loss_weight_list) != 3:
            raise ValueError(
                "when enable_cd, loss_weight_list must have length 3 "
                "[w_cls, w_bias, w_ind]"
            )
        w_cls, w_bias, w_ind = (
            loss_weight_list[0],
            loss_weight_list[1],
            loss_weight_list[2],
        )
    else:
        w_cls = loss_weight_list[0]

    det_model.train()
    if enable_cfi and cfi_pipeline is not None:
        cfi_pipeline.eval()

    metrics = metrics_accumulator()
    accum_count = 0
    use_bf16_amp = (
        device.type == "cuda"
        and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    )
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16_amp
        else nullcontext()
    )

    for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=desc, ncols=80)):
        videos = batch["video"].contiguous().to(device)
        base_labels = torch.tensor(
            [int(base_label) for base_label in batch["base_label"]],
            dtype=torch.long,
            device=device,
        )
        real_mask = base_labels == 0
        batch_size = videos.size(0)

        # Optionally generate counterfactual videos (inside AMP context; CFI under
        # no_grad) and run a single detector forward on [original, cf_videos].
        cf_videos = None
        with amp_ctx:
            if enable_cfi and cfi_pipeline is not None and real_mask.any():
                real_videos = videos[real_mask]
                real_indices = torch.where(real_mask)[0].cpu().tolist()
                captions = (
                    [batch["semantic_source_label"][i] for i in real_indices]
                    if "semantic_source_label" in batch
                    else [""] * len(real_indices)
                )
                with torch.no_grad():
                    cfi_pipeline_outputs = cfi_pipeline(
                        real_videos, captions=captions
                    )
                    cf_videos = cfi_pipeline_outputs["cf_videos"]
                    if batch_idx == 0:
                        try:
                            save_video_tensor_to_frames(
                                cfi_pipeline_outputs["real_videos"][0:1],
                                "./cfi_pipeline_outputs/real_videos",
                            )
                            save_video_tensor_to_frames(
                                cfi_pipeline_outputs["rec_videos"][0:1],
                                "./cfi_pipeline_outputs/rec_videos",
                            )
                            save_video_tensor_to_frames(
                                cfi_pipeline_outputs["far_videos"][0:1],
                                "./cfi_pipeline_outputs/far_videos",
                            )
                            save_video_tensor_to_frames(
                                cfi_pipeline_outputs["cf_videos"][0:1],
                                "./cfi_pipeline_outputs/cf_videos",
                            )
                        except Exception as exc:
                            logging.warning(
                                "CFI debug frame save failed (batch_idx=0): %s",
                                exc,
                            )

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
                outputs["causal_logits"] if enable_cd else outputs["logits"]
            )
            metrics.add("ori_base_labels", base_labels[:batch_size].cpu())
            metrics.add(
                "ori_preds",
                torch.sigmoid(main_logits[:batch_size, 0]).cpu().detach(),
            )
            if num_cf_videos > 0:
                metrics.add(
                    "cf_preds",
                    torch.sigmoid(main_logits[batch_size:, 0]).cpu().detach(),
                )

            if num_cf_videos > 0 and cf_cls_weight != 1.0:
                logits_binary = (
                    main_logits if main_logits.dim() == 1 else main_logits.squeeze(-1)
                )
                cls_targets = base_labels.float()
                cls_weights = torch.ones_like(cls_targets)
                cls_weights[batch_size:] = cf_cls_weight
                L_cls = F.binary_cross_entropy_with_logits(
                    logits_binary,
                    cls_targets,
                    weight=cls_weights,
                    reduction="sum",
                ) / cls_weights.sum()
            else:
                L_cls = classification_loss(main_logits, base_labels)
            metrics.add("cls_loss", L_cls.item())

            if enable_cd:
                bias_labels = base_labels.clone()
                if num_cf_videos > 0:
                    bias_labels[batch_size:] = 0
                L_bias = classification_loss(
                    outputs["non_causal_logits"], bias_labels
                )
                L_ind = independence_loss(
                    outputs["causal_features"],
                    outputs["non_causal_features"],
                )
                L_total = w_cls * L_cls + w_bias * L_bias + w_ind * L_ind
                metrics.add("bias_loss", L_bias.item())
                metrics.add("ind_loss", L_ind.item())
                metrics.add(
                    "bias_preds",
                    torch.sigmoid(
                        outputs["non_causal_logits"][:batch_size, 0]
                    ).cpu().detach(),
                )
                if num_cf_videos > 0:
                    metrics.add(
                        "bias_cf_preds",
                        torch.sigmoid(
                            outputs["non_causal_logits"][batch_size:, 0]
                        ).cpu().detach(),
                    )
            else:
                L_total = w_cls * L_cls

            metrics.add("total_loss", L_total.item())
            (L_total / grad_accum_steps).backward()

        accum_count += 1
        if accum_count >= grad_accum_steps:
            optimizer.step()
            optimizer.zero_grad()
            accum_count = 0

    # Trailing micro-batches when len(loader) % grad_accum_steps != 0.
    if accum_count > 0:
        optimizer.step()
        optimizer.zero_grad()

    ori_preds = metrics.get_concat_tensor("ori_preds").float().numpy()
    ori_true_labels = metrics.get_concat_tensor("ori_base_labels").float().numpy()
    ori_pred_labels = (ori_preds > 0.5).astype(int)
    train_acc = accuracy_score(ori_true_labels, ori_pred_labels)
    train_loss_components: Dict[str, float] = {
        "total_loss": metrics.get_mean("total_loss"),
        "cls_loss": metrics.get_mean("cls_loss"),
    }
    if enable_cd:
        train_loss_components["bias_loss"] = metrics.get_mean("bias_loss")
        train_loss_components["ind_loss"] = metrics.get_mean("ind_loss")

    train_results = {"train_acc": train_acc, "train_loss_components": train_loss_components}
    if metrics.exists("cf_preds"):
        cf_preds = metrics.get_concat_tensor("cf_preds").float().numpy()
        train_results["cf_acc"] = float((cf_preds > 0.5).mean())
    if metrics.exists("bias_preds"):
        bias_preds = metrics.get_concat_tensor("bias_preds").float().numpy()
        bias_pred_labels = (bias_preds > 0.5).astype(int)
        train_results["bias_acc"] = accuracy_score(ori_true_labels, bias_pred_labels)
    if metrics.exists("bias_cf_preds"):
        bias_cf_preds = metrics.get_concat_tensor("bias_cf_preds").float().numpy()
        train_results["bias_cf_acc"] = float(1.0 - (bias_cf_preds > 0.5).mean())
    return train_results
