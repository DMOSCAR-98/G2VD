"""Check whether the main G2VD runtime dependencies are importable."""

from __future__ import annotations

import importlib.util
import sys
from typing import Dict, List


REQUIRED_MODULES: Dict[str, str] = {
    "torch": "torch",
    "torchvision": "torchvision",
    "pytorchvideo": "pytorchvideo",
    "av": "av",
    "albumentations": "albumentations",
    "opencv-python": "cv2",
    "Pillow": "PIL",
    "omegaconf": "omegaconf",
    "PyYAML": "yaml",
    "numpy": "numpy",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "transformers": "transformers",
    "huggingface-hub": "huggingface_hub",
    "safetensors": "safetensors",
    "sentencepiece": "sentencepiece",
    "timm": "timm",
    "einops": "einops",
    "ftfy": "ftfy",
    "regex": "regex",
    "tqdm": "tqdm",
}

OPTIONAL_MODULES: Dict[str, str] = {
    "xformers": "xformers",
    "tensorboard": "tensorboard",
    "pytorch-lightning": "pytorch_lightning",
    "fairscale": "fairscale",
    "peft": "peft",
}


def _missing_modules(modules: Dict[str, str]) -> List[str]:
    missing: List[str] = []
    for package_name, module_name in modules.items():
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_name)
    return missing


def main() -> int:
    missing_required = _missing_modules(REQUIRED_MODULES)
    missing_optional = _missing_modules(OPTIONAL_MODULES)

    if missing_required:
        print("Missing required packages:")
        for name in missing_required:
            print(f"  - {name}")
    else:
        print("All required packages are importable.")

    if missing_optional:
        print("Missing optional packages:")
        for name in missing_optional:
            print(f"  - {name}")
    else:
        print("All optional packages are importable.")

    return 1 if missing_required else 0


if __name__ == "__main__":
    sys.exit(main())
