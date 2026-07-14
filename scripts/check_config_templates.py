"""Validate release YAML config templates without launching training."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in lightweight envs.
    yaml = None


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / "configs" / "templates"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        text = f.read()
    if yaml is None:
        return {"__raw_text__": text}
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a YAML mapping.")
    return data


def _raw_has_key(text: str, key: str) -> bool:
    return re.search(rf"(?m)^\s*{re.escape(key)}\s*:", text) is not None


def _require_keys(data: Dict[str, Any], keys: Iterable[str], path: Path) -> List[str]:
    errors: List[str] = []
    raw_text = data.get("__raw_text__")
    for key in keys:
        if raw_text is not None:
            has_key = _raw_has_key(str(raw_text), key)
        else:
            has_key = key in data
        if not has_key:
            errors.append(f"{path}: missing key `{key}`")
    return errors


def _check_train_template(path: Path, data: Dict[str, Any]) -> List[str]:
    errors = _require_keys(
        data,
        [
            "save_dir",
            "dataset_root",
            "train_metadata_dir",
            "train_video_data_list",
            "val_metadata_dir",
            "val_video_data_list",
            "det_model",
        ],
        path,
    )
    if "__raw_text__" in data:
        raw_text = str(data["__raw_text__"])
        for key in ["target", "params", "checkpoint_path", "video_backbone"]:
            if not _raw_has_key(raw_text, key):
                errors.append(f"{path}: missing key `{key}`")
        return errors
    det_model = data.get("det_model", {})
    if not isinstance(det_model, dict):
        errors.append(f"{path}: `det_model` must be a mapping")
        return errors
    errors.extend(_require_keys(det_model, ["target", "params", "checkpoint_path"], path))
    params = det_model.get("params", {})
    if not isinstance(params, dict) or "video_backbone" not in params:
        errors.append(f"{path}: missing `det_model.params.video_backbone`")
    return errors


def _check_test_template(path: Path, data: Dict[str, Any]) -> List[str]:
    return _require_keys(
        data,
        [
            "save_dir",
            "dataset_root",
            "test_metadata_dir",
            "test_video_data_list",
            "det_checkpoint_path",
        ],
        path,
    )


def main() -> int:
    if not TEMPLATE_DIR.exists():
        print(f"Template directory does not exist: {TEMPLATE_DIR}")
        return 1

    errors: List[str] = []
    templates = sorted(TEMPLATE_DIR.glob("*.yaml"))
    if not templates:
        print(f"No YAML templates found under {TEMPLATE_DIR}")
        return 1

    for path in templates:
        data = _load_yaml(path)
        if path.name.startswith("train_"):
            errors.extend(_check_train_template(path, data))
        elif path.name.startswith("test_"):
            errors.extend(_check_test_template(path, data))
        else:
            errors.append(f"{path}: unknown template type")

    if errors:
        print("Config template check failed:")
        for err in errors:
            print(f"  - {err}")
        return 1

    mode = "PyYAML" if yaml is not None else "fallback text scan"
    print(f"Validated {len(templates)} config templates ({mode}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
