"""
Load dataset metadata JSON files into labeled_video_paths tuples.

Each entry is (video_path, info_dict) with paths resolved against dataset_root.
"""
import json
import os
from typing import Any, Dict, List, Tuple, Union


def load_labeled_video_paths(
    json_paths: Union[str, List[str]],
    dataset_root: str,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Load one or more metadata files and return a list in labeled_video_paths format.

    Args:
        json_paths (Union[str, List[str]]): Single json file path (str) or list of
            json file paths (List[str]).
        dataset_root (str): Root directory path of the dataset.

    Returns:
        List[Tuple[str, Dict[str, Any]]]: Each tuple contains a video path and an
            information dictionary.
    """

    if isinstance(json_paths, str):
        json_paths = [json_paths]

    labeled_video_paths = []
    for json_path in json_paths:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Integrate data into specified format, only keep valid complete data
            labeled_video_paths.extend([
                (os.path.join(dataset_root, item["video_path"]), item["info_dict"])
                for item in data
                if item.get("video_path") and item.get("info_dict")
            ])

    return labeled_video_paths
