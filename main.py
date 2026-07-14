"""
Main training script for AI-generated video detection with optional CFI and CD stages.

Reads YAML config, loads data, instantiates detector, runs training. After merging CLI
overrides, save_dir is set from cfg save_dir (same pattern as test.py);
checkpoints and training_logs.txt are written there.
Video batches from dataloaders use BTCHW layout, shape [B, T, C, H, W].
CFI/CD enable when epoch progress strictly exceeds cfi_stage/cd_stage (progress > stage;
[0,1]). Default 1.0 means never enable. CFIPipeline is instantiated on first CFI
(not at startup) to reduce memory. If progress passes cfi_stage but the config
omits cfi_pipeline, training logs a warning and no counterfactuals are used.
On first CD enable: if detector already has CD classifier, no replace;
else replace_classifier; then re-create optimizer/scheduler.
Validation is intentionally run on original videos only (no CFI) for stable and
comparable checkpoints across epochs. Best checkpoint "best_det.pth" is selected by
strictly highest validation accuracy, with an additional guard val_acc < train_acc
to avoid selecting suspiciously optimistic validation states. Optional cfg
checkpoint_min_real_acc/checkpoint_min_fake_acc further reject imbalanced
validation states. ReduceLROnPlateau uses validation accuracy (mode=max) and runs
after each validation block; training
logs append LR for all param groups (g0, g1, ...) right after the epoch update,
before validation and scheduler.step, so values match the LR used during that epoch.
CD training requires CFI so cf_videos are in the batch; when CD is enabled
without CFI, this script raises a ValueError instead of continuing with a
misconfigured run. Without counterfactual videos, causal and non-causal
branches share the same inputs and binary labels and do not get the intended
causal vs non-causal split.
Default: joint training (backbone + CD classifier). Uncomment the freeze block in code
to freeze backbone and train only the CD classifier.
Mixed precision: on CUDA, bfloat16 autocast when torch.cuda.is_bf16_supported();
otherwise fp32. CPU uses fp32. No fp16 or GradScaler.
When CUDA is available, the detector is wrapped with DataParallel (all visible
GPUs; use CUDA_VISIBLE_DEVICES to limit). CFIPipeline is not wrapped with DP:
it uses VAEPool (models from vae_pool stay on CPU; one reconstructor moves
to device per forward), which is incompatible with DP; CFI runs on the main device only.
Baseline: enable_cfi=False, enable_cd=False, collect_data=False.

Optional cfg seed (int) or CLI dotlist seed=42: merged cfg drives set_global_seed
before dataloaders and model init; defaults to 42 if omitted.

Output layout: training_logs.txt, checkpoints, etc. under cfg save_dir (exact
subpaths are config-specific, e.g. train_results/<category>/<model>/.../).
When generate_tsne is enabled, main.py creates
save_dir/tsne_visualizations_during_training/ once; per-epoch PNG paths are
os.path.join(save_dir, "tsne_visualizations_during_training", <filename>).
See test.py module docstring for test output layout.
"""
import argparse
import logging
import os
import time
from typing import Any, Dict, Mapping

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from omegaconf import OmegaConf

from baseline_models.baseline_model_wrapper import BaselineModelWrapper
from data.dataloader import create_dataloaders
from my_utils import (
    convert_labeled_video_paths,
    data_sampler,
    dynamic_parameter_scheduler,
    get_inner_model,
    instantiate_from_config,
    load_labeled_video_paths,
    load_metadata_json_paths,
    save_model_checkpoint,
    set_global_seed,
    visualize_tsne,
)
from run_functions import eval_model, train_epoch


def _log_by_gs_metrics(
    title: str,
    by_gs: Mapping[Any, Dict[str, Any]],
    reverse_map: Mapping[str, str],
) -> None:
    """Logs one per-generation-source block (eval_model val_by_gs or val_bias_by_gs dict).

    Each entry lists gs_acc and gs_num_samples. Display name is
    reverse_map[str(gs id)] when truthy; otherwise real (convert_labeled_video_paths
    maps real to "" under key "-1", so empty string becomes real in the log).
    """
    if not by_gs:
        return
    logging.info(title)
    for gs_label in sorted(by_gs.keys(), key=lambda x: int(x)):
        gs_name = reverse_map.get(str(int(gs_label))) or "real"
        gs_results = by_gs[gs_label]
        logging.info(
            f"  {gs_name}: gs_acc={gs_results['gs_acc']:.2%}, "
            f"gs_num_samples={gs_results['gs_num_samples']}"
        )


def main() -> None:
    """Load config and data, instantiate models, run training with CFI/CD.

    Best checkpoint uses maximum validation accuracy; see module docstring.
    Checkpoints and training_logs.txt use save_dir from cfg after CLI merge
    (same pattern as test.py).
    """
    parser = argparse.ArgumentParser(description="Train ai-generated video detection")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config"
    )
    args, unknown = parser.parse_known_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    cfg = OmegaConf.load(args.config)
    if unknown:
        cli_cfg = OmegaConf.from_dotlist(unknown)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    seed = cfg.get("seed", 42)
    set_global_seed(seed)

    save_dir = cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "training_logs.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    logging.info(f"Device: {device}")
    if device.type != "cuda":
        logging.info("AMP: CPU path, using fp32 (no autocast).")
    elif getattr(torch.cuda, "is_bf16_supported", lambda: False)():
        logging.info("AMP: CUDA bfloat16 autocast enabled.")
    else:
        logging.info(
            "AMP: CUDA device does not support native bfloat16; using fp32."
        )
    logging.info(f"Config loaded from: {args.config}")
    logging.info(f"Config: {cfg}")

    # cfi_stage, cd_stage in [0, 1]; enable when epoch_progress > stage (strictly).
    # Default 1.0 => never enable CFI/CD (linear training only).
    cfi_stage = float(cfg.get("cfi_stage", 1.0))
    cd_stage = float(cfg.get("cd_stage", 1.0))
    train_epochs = cfg["train_epochs"]
    val_freq = cfg["val_freq"]
    grad_accum_steps = cfg["grad_accum_steps"]
    generate_tsne = cfg.get("generate_tsne", False)
    cf_cls_weight = float(cfg.get("cf_cls_weight", 1.0))
    # Start value for CD independence weight warm-up, e.g. 50 -> target w_ind=500.
    ind_weight_warmup_start = float(cfg.get("ind_weight_warmup_start", 0))
    ind_weight_warmup_epochs = int(cfg.get("ind_weight_warmup_epochs", 0))
    checkpoint_train_val_guard = cfg.get("checkpoint_train_val_guard", True)
    checkpoint_min_real_acc = float(cfg.get("checkpoint_min_real_acc", 0))
    checkpoint_min_fake_acc = float(cfg.get("checkpoint_min_fake_acc", 0))
    logging.info(
        "Training stability: cf_cls_weight=%s, "
        "ind_weight_warmup_start_weight=%s, ind_weight_warmup_epochs=%s, "
        "checkpoint_train_val_guard=%s, checkpoint_min_real_acc=%s, "
        "checkpoint_min_fake_acc=%s",
        cf_cls_weight,
        ind_weight_warmup_start,
        ind_weight_warmup_epochs,
        checkpoint_train_val_guard,
        checkpoint_min_real_acc,
        checkpoint_min_fake_acc,
    )

    # Load train data
    train_metadata_json_paths = load_metadata_json_paths(
        cfg["train_metadata_dir"],
        cfg["train_video_data_list"],
    )
    train_labeled_video_paths, train_gs_label_map = convert_labeled_video_paths(
        data_sampler(
            load_labeled_video_paths(
                train_metadata_json_paths,
                cfg["dataset_root"],
            ),
            cfg["train_sample_ratio"],
            same_semantic=False,
        ),
        return_map=True,
    )
    logging.info(
        f"Train size: {len(train_labeled_video_paths)}, "
        f"gs label map: {train_gs_label_map}"
    )

    # Load val data
    val_metadata_json_paths = load_metadata_json_paths(
        cfg["val_metadata_dir"],
        cfg["val_video_data_list"],
    )
    val_labeled_video_paths, val_gs_label_map = convert_labeled_video_paths(
        data_sampler(
            load_labeled_video_paths(
                val_metadata_json_paths,
                cfg["dataset_root"],
            ),
            cfg["val_sample_ratio"],
        ),
        return_map=True,
    )
    logging.info(
        f"Val size: {len(val_labeled_video_paths)}, "
        f"gs label map: {val_gs_label_map}"
    )

    train_dataloader, val_dataloader = create_dataloaders(
        cfg,
        train_labeled_video_paths,
        val_labeled_video_paths,
    )

    # Effective batch size after gradient accumulation (per optimizer step).
    train_batch_size = int(cfg["train_batch_size"])
    effective_batch_size = train_batch_size * grad_accum_steps
    logging.info(
        f"Gradient accumulation: {grad_accum_steps} steps, "
        f"train_batch_size={train_batch_size} -> effective_batch_size={effective_batch_size}"
    )

    # Detector
    det_model = instantiate_from_config(cfg["det_model"])
    det_model = det_model.to(device)

    # Baseline: no CFI, no CD, no t-SNE collection
    is_baseline = isinstance(det_model, BaselineModelWrapper)
    if is_baseline:
        logging.info("Baseline: no CFI/CD, no collect_data")
        generate_tsne = False

    if generate_tsne:
        os.makedirs(
            os.path.join(save_dir, "tsne_visualizations_during_training"),
            exist_ok=True,
        )

    # Optional checkpoint load (train YAML; e.g. G2VD staged paths under seed_${seed}).
    checkpoint_path = cfg["det_model"].get("checkpoint_path")
    if checkpoint_path and os.path.exists(checkpoint_path):
        logging.info(f"Loading detector checkpoint: {checkpoint_path}")
        # Checkpoint from this project's training; allow full unpickle (PyTorch 2.6+).
        ckpt = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        state_dict = ckpt.get("state_dict", ckpt)
        missing, unexpected = det_model.load_state_dict(
            state_dict, strict=False
        )
        if missing:
            logging.warning(f"Missing keys: {missing[:5]}...")
        if unexpected:
            logging.warning(f"Unexpected keys: {unexpected[:5]}...")
    elif checkpoint_path:
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )
    else:
        logging.info("No detector checkpoint, using init weights")

    # Wrap with DataParallel when CUDA available (unified single/multi-GPU).
    # Use CUDA_VISIBLE_DEVICES to limit which GPUs are used.
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        det_model = torch.nn.DataParallel(
            det_model, device_ids=list(range(num_gpus))
        )
        logging.info(
            f"Model wrapped with DataParallel (device_ids={list(range(num_gpus))})"
        )

    # Optimizer (re-created when first entering CD)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, det_model.parameters()),
        lr=float(cfg["det_model"]["base_learning_rate"]),
        weight_decay=cfg["weight_decay"],
    )
    # Maximize validation accuracy; scheduler.step receives that value each val epoch.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.1,
        patience=5,
        min_lr=1e-7,
    )

    # CFIPipeline instantiated on first CFI below (reduces memory when cfi_stage > 0)
    cfi_pipeline = None

    # loss_weight_list: length 3 [w_cls, w_bias, w_ind] when enable_cd (train_epoch).
    # When omitted, train_epoch defaults to [1.0, 1.0, 1.0] for CD.
    loss_weight_list = cfg.get("loss_weight_list", None)

    best_val_acc: float | None = None
    best_info = None
    enable_cfi_flag = False
    enable_cd_flag = False
    cd_epoch_idx = 0
    training_start_time = time.time()

    for epoch_idx in range(1, train_epochs + 1):
        # Epoch progress in (0, 1]; enable when progress > stage (so stage=1.0 never enables)
        progress = epoch_idx / train_epochs
        enable_cfi = (not is_baseline) and (progress > cfi_stage)
        enable_cd = (not is_baseline) and (progress > cd_stage)

        # Hard constraint: CD must be paired with CFI (counterfactual rows).
        if enable_cd and not enable_cfi:
            raise ValueError(
                "Invalid stage configuration: CD requires CFI, but current epoch "
                f"{epoch_idx} enables CD (cd_stage={cd_stage}) while CFI is off "
                f"(cfi_stage={cfi_stage}). Please ensure cfi_stage < cd_stage, or "
                "set cd_stage to 1.0 to disable CD."
            )
        if enable_cd and "cfi_pipeline" not in cfg:
            raise ValueError(
                "Invalid config: CD requires CFI, but 'cfi_pipeline' is missing "
                "from config. Add a cfi_pipeline block or set cd_stage to 1.0."
            )

        # First time entering CFI: instantiate CFIPipeline to reduce memory until CFI stage
        if enable_cfi and not enable_cfi_flag:
            enable_cfi_flag = True
            if "cfi_pipeline" in cfg:
                cfi_pipeline = instantiate_from_config(cfg["cfi_pipeline"])
                cfi_pipeline = cfi_pipeline.to(device)
                # CFIPipeline is not wrapped with DataParallel: VAEPool keeps all pool models on CPU
                # and moves only the selected VideoReconstructor to the input device per forward, which is
                # incompatible with DP (all params must be on device_ids[0]). CFI runs on
                # the main device only; detector remains DP-wrapped.
                logging.info("CFI enabled: CFIPipeline instantiated")
            else:
                logging.warning(
                    "Epoch progress exceeds cfi_stage but config has no 'cfi_pipeline' "
                    "entry; train_epoch runs with enable_cfi=True and no pipeline, so "
                    "no counterfactual videos are produced. Add a cfi_pipeline block "
                    "or set cfi_stage to 1.0 to disable CFI. Note: when CD is also "
                    "enabled, this is a hard error and training will stop."
                )

        # First time entering CD: replace classifier (if not already CD), then
        # re-create optimizer/scheduler. Default: joint training (backbone +
        # CD classifier). Uncomment the freeze block below to freeze backbone and
        # train only the CD classifier.
        if enable_cd and not enable_cd_flag:
            enable_cd_flag = True
            _det_model = get_inner_model(det_model)  # inner module for DP-safe attribute access
            if hasattr(_det_model, "replace_classifier"):
                if _det_model._classifier_type == "cd":
                    logging.info(
                        "CD enabled (classifier already CD), no replace needed"
                    )
                else:
                    # When None, detector uses {} and injects embed_dim/num_classes.
                    classifier_init_params = cfg["det_model"].get("params", {}).get(
                        "classifier_init_params"
                    )
                    _det_model.replace_classifier("cd", classifier_init_params)
                    logging.info("CD enabled: classifier replaced")

                # ---------- Optional: uncomment below to freeze backbone ----------
                # Default: joint training (backbone + CD classifier). Uncomment the
                # next 2 lines to freeze backbone and train only the CD classifier.
                # for p in _det_model.backbone.parameters():
                #     p.requires_grad = False
                # ------------------------------------------------------------------
                backbone_trainable = any(
                    p.requires_grad for p in _det_model.backbone.parameters()
                )
                if backbone_trainable:
                    logging.info(
                        "CD stage: backbone + CD classifier trained jointly"
                    )
                else:
                    logging.info(
                        "CD stage: backbone frozen, only CD classifier trainable"
                    )

                # Re-create optimizer/scheduler with learning rate groups:
                # backbone (if trainable) uses base_lr; CD classifier uses cd_lr.
                # Each non-empty group logs its lr when appended (see below).
                base_lr = float(cfg["det_model"]["base_learning_rate"])
                cd_lr = float(cfg["det_model"]["base_learning_rate"])
                backbone_params = [
                    p for p in _det_model.backbone.parameters() if p.requires_grad
                ]
                cd_params = [
                    p for p in _det_model.classifier.parameters() if p.requires_grad
                ]
                param_groups = []
                if backbone_params:
                    param_groups.append({"params": backbone_params, "lr": base_lr})
                    logging.info(
                        "CD stage: optimizer param group added (backbone, lr=%s).",
                        base_lr,
                    )
                if cd_params:
                    param_groups.append({"params": cd_params, "lr": cd_lr})
                    logging.info(
                        "CD stage: optimizer param group added "
                        "(CD classifier, lr=%s).",
                        cd_lr,
                    )
                if not param_groups:
                    logging.warning(
                        "CD stage: no trainable parameters for backbone or CD "
                        "classifier; optimizer param groups are empty."
                    )
                    raise RuntimeError(
                        "CD stage: cannot build AdamW with empty param_groups. "
                        "Check backbone_ft and the CD classifier."
                    )
                optimizer = torch.optim.AdamW(
                    param_groups,
                    weight_decay=cfg["weight_decay"],
                )
                # Same as pre-CD scheduler: maximize validation accuracy each val epoch.
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="max",
                    factor=0.1,
                    patience=5,
                    min_lr=1e-7,
                )
                logging.info(
                    "CD stage: optimizer and scheduler re-created."
                )

        effective_loss_weight_list = loss_weight_list
        if enable_cd:
            cd_epoch_idx += 1
            if loss_weight_list is None:
                effective_loss_weight_list = [1.0, 1.0, 1.0]
            else:
                effective_loss_weight_list = list(loss_weight_list)
            if (
                ind_weight_warmup_epochs > 0
                and len(effective_loss_weight_list) >= 3
            ):
                target_w_ind = float(effective_loss_weight_list[2])
                warmup_step = min(cd_epoch_idx, ind_weight_warmup_epochs)
                effective_w_ind = dynamic_parameter_scheduler(
                    strategy="linear",
                    step=warmup_step,
                    total_steps=ind_weight_warmup_epochs,
                    initial_value=ind_weight_warmup_start,
                    final_value=target_w_ind,
                )
                effective_loss_weight_list[2] = effective_w_ind
            logging.info(
                "Effective CD loss weights: %s (cd_epoch=%s)",
                effective_loss_weight_list,
                cd_epoch_idx,
            )

        # Train one epoch
        train_results = train_epoch(
            det_model,
            train_dataloader,
            optimizer,
            device,
            grad_accum_steps=grad_accum_steps,
            enable_cfi=enable_cfi,
            enable_cd=enable_cd,
            cfi_pipeline=cfi_pipeline,
            loss_weight_list=effective_loss_weight_list,
            cf_cls_weight=cf_cls_weight,
            desc=f"Train Epoch {epoch_idx}/{train_epochs}",
        )

        train_loss_components = train_results["train_loss_components"]
        train_loss_str = ", ".join(f"{k}: {v:.4f}" for k, v in train_loss_components.items())
        log_msg = (
            f"Train Epoch {epoch_idx}/{train_epochs} - "
            f"Train Acc: {train_results['train_acc']:.2%}, {train_loss_str}"
        )
        if "cf_acc" in train_results:
            log_msg += f", CF Acc: {train_results['cf_acc']:.2%}"
        if "bias_acc" in train_results:
            log_msg += f", Bias Acc: {train_results['bias_acc']:.2%}"
        if "bias_cf_acc" in train_results:
            log_msg += f", Bias CF Acc: {train_results['bias_cf_acc']:.2%}"
        # All param groups (g0, g1, ...); LR before validation and scheduler.step.
        lr_str = ", ".join(
            f"g{i}:{pg['lr']:.2e}" for i, pg in enumerate(optimizer.param_groups)
        )
        log_msg += f", LR: {lr_str}"
        logging.info(log_msg)

        # Validation
        if epoch_idx % val_freq == 0:
            # Original-only val (no CFI): metrics comparable across epochs for best_det and LR schedule.
            val_results = eval_model(
                det_model,
                val_dataloader,
                device,
                return_by_gs=True,
                collect_data=generate_tsne,
                cfi_pipeline=None,
                desc=f"Val Epoch {epoch_idx}",
            )

            scheduler.step(val_results["val_acc"])

            # val_* metrics may be None depending on class coverage; guard logs.
            val_real_acc_str = (
                f"{val_results['val_real_acc']:.2%}"
                if val_results["val_real_acc"] is not None
                else "N/A"
            )
            val_fake_acc_str = (
                f"{val_results['val_fake_acc']:.2%}"
                if val_results["val_fake_acc"] is not None
                else "N/A"
            )
            val_auc_str = (
                f"{val_results['val_auc']:.4f}"
                if val_results["val_auc"] is not None
                else "N/A"
            )
            val_ap_str = (
                f"{val_results['val_ap']:.4f}"
                if val_results["val_ap"] is not None
                else "N/A"
            )
            val_f1_str = (
                f"{val_results['val_f1']:.4f}"
                if val_results["val_f1"] is not None
                else "N/A"
            )
            logging.info(
                f"Val Epoch {epoch_idx} - "
                f"Main Head: "
                f"Val Acc: {val_results['val_acc']:.2%}, "
                f"Val Real Acc: {val_real_acc_str}, "
                f"Val Fake Acc: {val_fake_acc_str}, "
                f"Val Loss: {val_results['val_loss']:.4f}, "
                f"Val AUC: {val_auc_str}, "
                f"Val AP: {val_ap_str}, "
                f"Val F1: {val_f1_str}"
            )

            reverse_map_gs = {v: k for k, v in val_gs_label_map.items()}
            _log_by_gs_metrics(
                "Val — Main Head, by generation source:",
                val_results.get("val_by_gs") or {},
                reverse_map_gs,
            )

            if "val_bias_acc" in val_results:
                val_bias_auc_str = (
                    f"{val_results['val_bias_auc']:.4f}"
                    if val_results["val_bias_auc"] is not None
                    else "N/A"
                )
                val_bias_ap_str = (
                    f"{val_results['val_bias_ap']:.4f}"
                    if val_results["val_bias_ap"] is not None
                    else "N/A"
                )
                val_bias_f1_str = (
                    f"{val_results['val_bias_f1']:.4f}"
                    if val_results["val_bias_f1"] is not None
                    else "N/A"
                )
                val_bias_real_acc_str = (
                    f"{val_results['val_bias_real_acc']:.2%}"
                    if val_results["val_bias_real_acc"] is not None
                    else "N/A"
                )
                val_bias_fake_acc_str = (
                    f"{val_results['val_bias_fake_acc']:.2%}"
                    if val_results["val_bias_fake_acc"] is not None
                    else "N/A"
                )
                logging.info(
                    f"Val Epoch {epoch_idx} - "
                    f"Bias Head: "
                    f"Val Bias Acc: {val_results['val_bias_acc']:.2%}, "
                    f"Val Bias Real Acc: {val_bias_real_acc_str}, "
                    f"Val Bias Fake Acc: {val_bias_fake_acc_str}, "
                    f"Val Bias Loss: {val_results['val_bias_loss']:.4f}, "
                    f"Val Bias AUC: {val_bias_auc_str}, "
                    f"Val Bias AP: {val_bias_ap_str}, "
                    f"Val Bias F1: {val_bias_f1_str}"
                )
                _log_by_gs_metrics(
                    "Val — Bias Head, by generation source:",
                    val_results.get("val_bias_by_gs") or {},
                    reverse_map_gs,
                )

            if "val_cf_acc" in val_results:
                logging.info(f"Val CF Acc: {val_results['val_cf_acc']:.2%}")
            if "val_bias_cf_acc" in val_results:
                logging.info(
                    f"Val Bias CF Acc: {val_results['val_bias_cf_acc']:.2%}"
                )

            # Standardized t-SNE (causal head and optional non-causal head when CD).
            # Causal features use base real/fake labels; non-causal features use
            # bias_labels because the non-causal head is optimized against them.
            # Full PNG paths:
            # vis_save_dir and nc_vis_save_dir; title= inline on visualize_tsne calls.
            # tsne_visualizations_during_training/ mkdir once at training startup.
            if generate_tsne and "collected_features" in val_results:
                filename = f"tsne_epoch_{epoch_idx:04d}.png"
                vis_save_dir = os.path.join(
                    save_dir, "tsne_visualizations_during_training", filename
                )
                _, _, tsne_metrics = visualize_tsne(
                    features=val_results["collected_features"],
                    labels=val_results["collected_base_labels"],
                    save_path=vis_save_dir,
                    title=f"t-SNE (real/fake) Epoch {epoch_idx}",
                    perplexity=40,
                    return_metrics=True,
                )
                tsne_silhouette = tsne_metrics["silhouette_score"]
                tsne_silhouette_str = (
                    f"{tsne_silhouette:.4f}"
                    if tsne_silhouette is not None
                    else "N/A"
                )
                logging.info(
                    f"Saved standardized t-SNE: {vis_save_dir}, "
                    f"silhouette={tsne_silhouette_str}, "
                    f"reduction={tsne_metrics['reduction']}"
                )
                if "collected_non_causal_features" in val_results:
                    nc_filename = (
                        f"tsne_non_causal_epoch_{epoch_idx:04d}.png"
                    )
                    nc_vis_save_dir = os.path.join(
                        save_dir,
                        "tsne_visualizations_during_training",
                        nc_filename,
                    )
                    _, _, nc_tsne_metrics = visualize_tsne(
                        features=val_results["collected_non_causal_features"],
                        labels=val_results["collected_bias_labels"],
                        save_path=nc_vis_save_dir,
                        title=f"t-SNE (non-causal, real/fake) Epoch {epoch_idx}",
                        perplexity=40,
                        return_metrics=True,
                    )
                    nc_tsne_silhouette = nc_tsne_metrics["silhouette_score"]
                    nc_tsne_silhouette_str = (
                        f"{nc_tsne_silhouette:.4f}"
                        if nc_tsne_silhouette is not None
                        else "N/A"
                    )
                    logging.info(
                        f"Saved standardized t-SNE (non-causal): {nc_vis_save_dir}, "
                        f"silhouette={nc_tsne_silhouette_str}, "
                        f"reduction={nc_tsne_metrics['reduction']}"
                    )

            # Best checkpoint: strictly highest val_acc, plus configured guards.
            is_best_val = (
                best_val_acc is None or val_results["val_acc"] > best_val_acc
            )
            pass_train_val_guard = (
                (not checkpoint_train_val_guard)
                or val_results["val_acc"] < train_results["train_acc"]
            )
            pass_real_guard = (
                checkpoint_min_real_acc <= 0
                or (
                    val_results["val_real_acc"] is not None
                    and val_results["val_real_acc"] >= checkpoint_min_real_acc
                )
            )
            pass_fake_guard = (
                checkpoint_min_fake_acc <= 0
                or (
                    val_results["val_fake_acc"] is not None
                    and val_results["val_fake_acc"] >= checkpoint_min_fake_acc
                )
            )
            if (
                is_best_val
                and pass_train_val_guard
                and pass_real_guard
                and pass_fake_guard
            ):
                best_val_acc = val_results["val_acc"]
                # Reuse this epoch's formatted strings for best_info (Main Head vs Bias Head).
                best_auc_str = val_auc_str
                best_ap_str = val_ap_str
                best_f1_str = val_f1_str
                best_info = (
                    f"Best Epoch (by Val Acc): {epoch_idx} - "
                    f"Train Acc: {train_results['train_acc']:.2%}, "
                    f"Main Head: "
                    f"Val Acc: {best_val_acc:.2%}, "
                    f"Val Real Acc: {val_real_acc_str}, "
                    f"Val Fake Acc: {val_fake_acc_str}, "
                    f"Val Loss: {val_results['val_loss']:.4f}, "
                    f"Val AUC: {best_auc_str}, "
                    f"Val AP: {best_ap_str}, "
                    f"Val F1: {best_f1_str}"
                )
                if "val_cf_acc" in val_results:
                    best_info += f", Val CF Acc: {val_results['val_cf_acc']:.2%}"
                if "val_bias_acc" in val_results:
                    best_bias_auc_str = val_bias_auc_str
                    best_bias_ap_str = val_bias_ap_str
                    best_bias_f1_str = val_bias_f1_str
                    best_info += (
                        f", Bias Head: "
                        f"Val Bias Acc: {val_results['val_bias_acc']:.2%}, "
                        f"Val Bias Real Acc: {val_bias_real_acc_str}, "
                        f"Val Bias Fake Acc: {val_bias_fake_acc_str}, "
                        f"Val Bias Loss: {val_results['val_bias_loss']:.4f}, "
                        f"Val Bias AUC: {best_bias_auc_str}, "
                        f"Val Bias AP: {best_bias_ap_str}, "
                        f"Val Bias F1: {best_bias_f1_str}"
                    )
                if "val_bias_cf_acc" in val_results:
                    best_info += (
                        f", Val Bias CF Acc: {val_results['val_bias_cf_acc']:.2%}"
                    )
                logging.info(best_info)
                best_path = os.path.join(save_dir, "best_det.pth")
                save_model_checkpoint(
                    get_inner_model(det_model),
                    best_path,
                    is_lora=False,
                    extra={
                        "epoch": epoch_idx,
                        "train_acc": train_results["train_acc"],
                        "val_acc": val_results["val_acc"],
                        "val_auc": val_results["val_auc"],
                        "init_params": get_inner_model(det_model).get_init_params(),
                    },
                )
                logging.info(f"Saved best detector: {best_path}")
            else:
                reasons = []
                if not is_best_val:
                    prev_best_str = (
                        f"{best_val_acc:.2%}" if best_val_acc is not None else "N/A"
                    )
                    reasons.append(
                        "not strictly better than best "
                        f"(val_acc={val_results['val_acc']:.2%}, "
                        f"best_val_acc={prev_best_str})"
                    )
                if not pass_train_val_guard:
                    reasons.append(
                        "train-val guard failed "
                        f"(val_acc={val_results['val_acc']:.2%}, "
                        f"train_acc={train_results['train_acc']:.2%}, "
                        "require val_acc < train_acc)"
                    )
                if not pass_real_guard:
                    reasons.append(
                        "real-acc guard failed "
                        f"(val_real_acc={val_real_acc_str}, "
                        f"require >= {checkpoint_min_real_acc:.2%})"
                    )
                if not pass_fake_guard:
                    reasons.append(
                        "fake-acc guard failed "
                        f"(val_fake_acc={val_fake_acc_str}, "
                        f"require >= {checkpoint_min_fake_acc:.2%})"
                    )
                logging.info(
                    "Best checkpoint not updated: " + "; ".join(reasons) + "."
                )

            # Latest checkpoint
            latest_path = os.path.join(save_dir, "latest_det.pth")
            save_model_checkpoint(
                get_inner_model(det_model),
                latest_path,
                is_lora=False,
                extra={
                    "epoch": epoch_idx,
                    "train_acc": train_results["train_acc"],
                    "val_acc": val_results["val_acc"],
                    "val_auc": val_results["val_auc"],
                    "init_params": get_inner_model(det_model).get_init_params(),
                },
            )
            logging.info(f"Saved latest detector: {latest_path}")

    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    total_hours, remainder = divmod(total_training_time, 3600)
    total_minutes, total_seconds = divmod(remainder, 60)
    logging.info(
        f"Total training time: "
        f"{int(total_hours)}h {int(total_minutes)}m {int(total_seconds)}s"
    )
    if best_info is not None:
        logging.info(best_info)


if __name__ == "__main__":
    main()
