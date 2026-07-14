"""
Resolve metadata.json paths under a dataset metadata directory.

Used when building training lists from named dataset folders (e.g. modelscope, show1).
"""
import os
from typing import List


def load_metadata_json_paths(
    metadata_dir: str,
    video_data_list: List[str],
) -> List[str]:
    """
    Load metadata.json file paths for specified video data.

    Args:
        metadata_dir (str): Metadata directory path, e.g.,
            "./dataset_metadata/gvf-2.8k/".
        video_data_list (List[str]): List of folder names, e.g.,
            ["modelscope", "show1"].

    Returns:
        List[str]: Complete list of json file paths.
    """
    return [
        os.path.join(metadata_dir, f"{video_data}_metadata.json")
        for video_data in video_data_list
    ]
