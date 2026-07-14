"""
Normalize generation_source_label in labeled video metadata to string indices.

Real maps to "-1"; fake sources map to "0", "1", ... for per-source metrics and t-SNE.
"""
from typing import Any, Dict, List, Tuple, Union


def convert_labeled_video_paths(
    labeled_video_paths: List[Tuple[str, Dict[str, Any]]],
    return_map: bool = False,
) -> Union[
    List[Tuple[str, Dict[str, Any]]],
    Tuple[List[Tuple[str, Dict[str, Any]]], Dict[str, str]],
]:
    """
    Convert generation_source_label to string numeric indices. Real ("")
    maps to "-1"; fake sources to "0","1","2",... for per-source eval/t-SNE.

    Args:
        labeled_video_paths (List[Tuple[str, Dict[str, Any]]]): List containing
            video paths and information dictionaries.
        return_map (bool): Whether to return the gs label mapping dictionary.

    Returns:
        Union[
            List[Tuple[str, Dict[str, Any]]],
            Tuple[List[Tuple[str, Dict[str, Any]]], Dict[str, str]]
        ]:
            If return_map is False, returns converted paths only.
            If return_map is True, returns (converted_paths, gs_label_map).

    Note:
        gs label mapping scheme:
        - Real videos (""): "-1"
        - Fake videos: "0", "1", "2", ... (by sorted generation source names)
    """

    # Collect all generation_source_label values and sorted unique gs labels
    gs_labels = [
        item[1]["generation_source_label"] for item in labeled_video_paths
    ]
    unique_gs_labels = sorted(
        set(gs_label for gs_label in gs_labels if gs_label != "")
    )

    # Build gs_label_map and convert paths: Real "" -> "-1"; fake -> "0","1","2",...
    gs_label_map: Dict[str, str] = {"": "-1"}
    for idx, gs_label in enumerate(unique_gs_labels, start=0):
        gs_label_map[gs_label] = str(idx)
    converted_paths = [
        (path, {**info, "generation_source_label": gs_label_map[info["generation_source_label"]]})
        for path, info in labeled_video_paths
    ]

    if return_map:
        return converted_paths, gs_label_map

    return converted_paths
