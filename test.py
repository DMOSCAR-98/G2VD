"""
Test script for evaluating trained ai-generated video detection models.

Loads trained detector (and optional CFIPipeline) from checkpoint and evaluates
on test data. Baseline vs VideoDetector is inferred before DataParallel (same as
main.py). Mixed precision: on CUDA, bfloat16 autocast when torch.cuda.is_bf16_supported();
otherwise fp32. CPU uses fp32. No fp16 or GradScaler.
When CUDA is available, only the detector is wrapped with DataParallel
(same device_ids as training). CFIPipeline runs on the main device only (not
DP-wrapped; see main.py).

Test loader batches use BTCHW layout, shape [B, T, C, H, W].

Metrics are computed via eval_model and returned under val_* keys (e.g. val_bias_acc,
val_bias_loss when CD), while this script logs human-readable lines with Test * prefixes
(e.g. Test Bias Acc, Test Bias Loss for the Bias Head, without repeating Val in the label).
Optional val_by_gs and val_bias_by_gs aggregates (per gs id, including -1 for real: gs_acc,
gs_num_samples) are logged when available, with separate headers for the Main Head vs
the Bias Head when CD is enabled.

Supports t-SNE visualization (disabled for baseline). Logs config path and
merged OmegaConf to test_logs.txt under save_dir from cfg after CLI merge
(same pattern as main.py).

Optional cfg seed (int) or CLI dotlist seed=42: merged cfg drives set_global_seed
before dataloaders and model init; defaults to 42 if omitted.

Output layout: artifacts under cfg save_dir (exact subpaths are config-specific,
e.g. test_results/ideal/<category>/<model>/.../<dataset_slug>/). Includes
test_logs.txt; metrics are logged there. When generate_tsne is enabled,
test.py creates save_dir/tsne_visualizations/ before saving causal tsne.png
and optional non-causal tsne_non_causal.png (titles passed inline to visualize_tsne
like main.py; test filenames omit epoch). See main.py for train output layout.
"""
import argparse
import logging
import os
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
    instantiate_from_config,
    load_labeled_video_paths,
    load_metadata_json_paths,
    set_global_seed,
    visualize_tsne,
)
from run_functions import eval_model


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
    """
    Load detector (and optional CFIPipeline) from checkpoints, evaluate on test
    set, and optionally generate t-SNE visualizations.

    Model type is inferred from checkpoint init_params: BaselineModelWrapper
    (model_name in params) vs VideoDetector. Baseline is detected before
    DataParallel via isinstance(det_model, BaselineModelWrapper), matching
    main.py. Baseline models force collect_data=False and generate_tsne=False.
    Writes merged config (after CLI overrides) to the log under save_dir from cfg,
    matching main.py.
    """
    parser = argparse.ArgumentParser(description="Test ai-generated video detection")
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
    log_file = os.path.join(save_dir, "test_logs.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
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

    # Load test data
    test_metadata_json_paths = load_metadata_json_paths(
        cfg["test_metadata_dir"],
        cfg["test_video_data_list"],
    )
    test_labeled_video_paths, test_gs_label_map = convert_labeled_video_paths(
        data_sampler(
            load_labeled_video_paths(
                test_metadata_json_paths,
                cfg["dataset_root"],
            ),
            cfg["test_sample_ratio"],
        ),
        return_map=True,
    )
    logging.info(
        f"Test size: {len(test_labeled_video_paths)}, "
        f"gs label map: {test_gs_label_map}"
    )

    # Map test_* keys to val_* for create_dataloaders (required; missing keys raise).
    # test_augmentation is optional; create_dataloaders defaults to resize+normalize.
    test_cfg = OmegaConf.create(dict(cfg))
    test_cfg["val_resolution"] = cfg["test_resolution"]
    test_cfg["val_num_samples"] = cfg["test_num_samples"]
    test_cfg["val_clip_duration"] = cfg["test_clip_duration"]
    test_cfg["val_batch_size"] = cfg["test_batch_size"]
    test_cfg["val_num_workers"] = cfg["test_num_workers"]
    if "test_augmentation" in cfg:
        test_cfg["test_augmentation"] = cfg["test_augmentation"]

    _, test_dataloader = create_dataloaders(
        test_cfg,
        train_labeled_video_paths=None,
        val_labeled_video_paths=test_labeled_video_paths,
    )

    # Load detector checkpoint
    det_ckpt_path = cfg["det_checkpoint_path"]
    if not os.path.exists(det_ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {det_ckpt_path}")
    logging.info(f"Loading detector from: {det_ckpt_path}")
    # Checkpoint from this project's training; allow full unpickle (PyTorch 2.6+).
    det_ckpt = torch.load(
        det_ckpt_path, map_location=device, weights_only=False
    )
    if "init_params" not in det_ckpt:
        raise KeyError(
            "Checkpoint missing 'init_params'. "
            "Use a checkpoint saved by the current training code."
        )
    det_init_params = det_ckpt["init_params"].copy()

    # Instantiate detector from init_params (baseline vs VideoDetector)
    if "model_name" in det_init_params:
        det_model_config = {
            "target": (
                "baseline_models.baseline_model_wrapper.BaselineModelWrapper"
            ),
            "params": det_init_params,
        }
        logging.info(f"Baseline model: {det_init_params['model_name']}")
    else:
        det_model_config = {
            "target": "modules.detector.VideoDetector",
            "params": det_init_params,
        }
        logging.info(
            f"VideoDetector: backbone={det_init_params.get('video_backbone')}, "
            f"classifier_type={det_init_params.get('classifier_type')}"
        )

    det_model = instantiate_from_config(det_model_config)
    det_model = det_model.to(device)
    det_state_dict = det_ckpt.get("state_dict", det_ckpt)
    missing, unexpected = det_model.load_state_dict(
        det_state_dict, strict=False
    )
    if missing:
        logging.warning(f"Missing keys: {missing[:5]}...")
    if unexpected:
        logging.warning(f"Unexpected keys: {unexpected[:5]}...")

    generate_tsne = cfg.get("generate_tsne", False)
    is_baseline = isinstance(det_model, BaselineModelWrapper)
    if is_baseline:
        logging.info("Baseline: no CFI/CD, no collect_data")
        generate_tsne = False

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        det_model = torch.nn.DataParallel(
            det_model, device_ids=list(range(num_gpus))
        )
        logging.info(
            f"Detector wrapped with DataParallel (device_ids={list(range(num_gpus))})"
        )
    logging.info("Detector loaded")

    # Optional CFIPipeline for evaluation on original + counterfactual videos.
    cfi_pipeline = None
    if not is_baseline and "cfi_pipeline" in cfg:
        cfi_pipeline = instantiate_from_config(cfg["cfi_pipeline"])
        cfi_pipeline = cfi_pipeline.to(device)
        # CFIPipeline is not wrapped with DataParallel (VAEPool keeps pool models on CPU;
        # see main.py). It runs on the main device only.
        logging.info("CFIPipeline instantiated")

    # Run evaluation
    logging.info("=" * 80)
    logging.info("Running evaluation on test set...")
    if cfi_pipeline is not None:
        logging.info("Evaluating on original + counterfactual videos")
    else:
        logging.info("Evaluating on original videos only")
    if cfg.get("test_augmentation"):
        logging.info("Test augmentation (non-ideal conditions) enabled")
    else:
        logging.info("Test augmentation disabled (ideal conditions)")
    logging.info("=" * 80)

    test_results = eval_model(
        det_model,
        test_dataloader,
        device,
        return_by_gs=True,
        collect_data=generate_tsne,
        cfi_pipeline=cfi_pipeline,
        desc="Testing",
    )

    # Log results
    logging.info("=" * 80)
    logging.info("Test Results:")
    logging.info("=" * 80)
    # eval_model keys use val_* (e.g. val_cf_acc, val_bias_*); this script logs Test * labels
    # (Test Bias * for the Bias Head) without repeating Val in the message text.
    # val_* metrics may be None depending on class coverage; guard logs with N/A strings.
    test_real_acc_str = (
        f"{test_results['val_real_acc']:.2%}"
        if test_results["val_real_acc"] is not None
        else "N/A"
    )
    test_fake_acc_str = (
        f"{test_results['val_fake_acc']:.2%}"
        if test_results["val_fake_acc"] is not None
        else "N/A"
    )
    test_auc_str = (
        f"{test_results['val_auc']:.4f}"
        if test_results["val_auc"] is not None
        else "N/A"
    )
    test_ap_str = (
        f"{test_results['val_ap']:.4f}"
        if test_results["val_ap"] is not None
        else "N/A"
    )
    test_f1_str = (
        f"{test_results['val_f1']:.4f}"
        if test_results["val_f1"] is not None
        else "N/A"
    )
    logging.info(
        f"Test Main Head: "
        f"Test Acc: {test_results['val_acc']:.2%}, "
        f"Test Real Acc: {test_real_acc_str}, "
        f"Test Fake Acc: {test_fake_acc_str}, "
        f"Test Loss: {test_results['val_loss']:.4f}, "
        f"Test AUC: {test_auc_str}, "
        f"Test AP: {test_ap_str}, "
        f"Test F1: {test_f1_str}"
    )

    reverse_map_gs = {v: k for k, v in test_gs_label_map.items()}
    _log_by_gs_metrics(
        "Test — Main Head, by generation source:",
        test_results.get("val_by_gs") or {},
        reverse_map_gs,
    )

    if "val_bias_acc" in test_results:
        # val_bias_* metrics may be None depending on class coverage; guard logs with N/A strings.
        test_bias_real_acc_str = (
            f"{test_results['val_bias_real_acc']:.2%}"
            if test_results["val_bias_real_acc"] is not None
            else "N/A"
        )
        test_bias_fake_acc_str = (
            f"{test_results['val_bias_fake_acc']:.2%}"
            if test_results["val_bias_fake_acc"] is not None
            else "N/A"
        )
        test_bias_auc_str = (
            f"{test_results['val_bias_auc']:.4f}"
            if test_results["val_bias_auc"] is not None
            else "N/A"
        )
        test_bias_ap_str = (
            f"{test_results['val_bias_ap']:.4f}"
            if test_results["val_bias_ap"] is not None
            else "N/A"
        )
        test_bias_f1_str = (
            f"{test_results['val_bias_f1']:.4f}"
            if test_results["val_bias_f1"] is not None
            else "N/A"
        )
        logging.info(
            f"Test Bias Head: "
            f"Test Bias Acc: {test_results['val_bias_acc']:.2%}, "
            f"Test Bias Real Acc: {test_bias_real_acc_str}, "
            f"Test Bias Fake Acc: {test_bias_fake_acc_str}, "
            f"Test Bias Loss: {test_results['val_bias_loss']:.4f}, "
            f"Test Bias AUC: {test_bias_auc_str}, "
            f"Test Bias AP: {test_bias_ap_str}, "
            f"Test Bias F1: {test_bias_f1_str}"
        )
        _log_by_gs_metrics(
            "Test — Bias Head, by generation source:",
            test_results.get("val_bias_by_gs") or {},
            reverse_map_gs,
        )

    if "val_cf_acc" in test_results:
        logging.info(f"Test CF Acc: {test_results['val_cf_acc']:.2%}")
    if "val_bias_cf_acc" in test_results:
        logging.info(
            f"Test Bias CF Acc: {test_results['val_bias_cf_acc']:.2%}"
        )

    # Standardized t-SNE (causal head and optional non-causal head when CD).
    # Causal features use base real/fake labels; non-causal features use
    # bias_labels because the non-causal head is optimized against them.
    if generate_tsne and "collected_features" in test_results:
        os.makedirs(os.path.join(save_dir, "tsne_visualizations"), exist_ok=True)
        filename = "tsne.png"
        vis_save_dir = os.path.join(save_dir, "tsne_visualizations", filename)
        _, _, tsne_metrics = visualize_tsne(
            features=test_results["collected_features"],
            labels=test_results["collected_base_labels"],
            save_path=vis_save_dir,
            title="t-SNE (real/fake)",
            perplexity=40,
            return_metrics=True,
        )
        tsne_silhouette = tsne_metrics["silhouette_score"]
        tsne_silhouette_str = (
            f"{tsne_silhouette:.4f}" if tsne_silhouette is not None else "N/A"
        )
        logging.info(
            f"Saved standardized t-SNE: {vis_save_dir}, "
            f"silhouette={tsne_silhouette_str}, "
            f"reduction={tsne_metrics['reduction']}"
        )
        if "collected_non_causal_features" in test_results:
            nc_filename = "tsne_non_causal.png"
            nc_vis_save_dir = os.path.join(
                save_dir, "tsne_visualizations", nc_filename
            )
            _, _, nc_tsne_metrics = visualize_tsne(
                features=test_results["collected_non_causal_features"],
                labels=test_results["collected_bias_labels"],
                save_path=nc_vis_save_dir,
                title="t-SNE (non-causal, real/fake)",
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

    logging.info("=" * 80)
    logging.info("Test completed successfully!")
    logging.info(f"Results saved to: {save_dir}")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
