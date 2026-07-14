"""
Data sampling utilities for train/val splits.

Label convention: 0 = real, 1 = fake (ai-generated). base_label "0" = real,
base_label "1" = fake (ai-generated) videos.
"""
import logging
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple


def data_sampler(
    labeled_video_paths: List[Tuple[str, Dict[str, Any]]],
    sample_ratio: float,
    same_semantic: bool = False,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Data sampling function that extracts data from labeled_video_paths
    according to specified strategy.

    Args:
        labeled_video_paths (List[Tuple[str, Dict[str, Any]]]): List of video
            metadata, where each element is a tuple of (video_path, info_dict).
        sample_ratio (float): Sampling ratio (0.0-1.0).
        same_semantic (bool): Maintain semantic consistency in sampling.

    Returns:
        List[Tuple[str, Dict[str, Any]]]: Sampled video metadata list with the
            same format as input.
    """
    if not labeled_video_paths:
        return []

    real_videos, fake_videos = [], []
    for video_path, video_info in labeled_video_paths:
        if video_info["base_label"] == "0":
            real_videos.append((video_path, video_info))
        elif video_info["base_label"] == "1":
            fake_videos.append((video_path, video_info))

    logging.info(
        f"Data sampling - Real videos: {len(real_videos)}, "
        f"Fake videos: {len(fake_videos)}"
    )

    if same_semantic:
        return _semantic_aware_sampling(
            real_videos, fake_videos, sample_ratio
        )
    else:
        return _random_sampling(real_videos, fake_videos, sample_ratio)


def _semantic_aware_sampling(
    real_videos: List[Tuple[str, Dict[str, Any]]],
    fake_videos: List[Tuple[str, Dict[str, Any]]],
    sample_ratio: float,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Perform semantic-aware sampling.

    Sample specified ratio from real videos, then for each sampled real video,
    find fake videos with the same semantic_source_label, ensuring no
    duplicate fake videos.

    Args:
        real_videos (List[Tuple[str, Dict[str, Any]]]): List of real video
            metadata tuples.
        fake_videos (List[Tuple[str, Dict[str, Any]]]): List of fake (ai-generated)
            video metadata tuples.
        sample_ratio (float): Sampling ratio (0.0-1.0).

    Returns:
        List[Tuple[str, Dict[str, Any]]]: Sampled video metadata list.
    """
    num_real_to_sample = max(1, int(len(real_videos) * sample_ratio))
    sampled_real_videos = random.sample(
        real_videos, min(num_real_to_sample, len(real_videos))
    )

    logging.info(
        f"Semantic-aware sampling - Sampling {len(sampled_real_videos)} from "
        f"{len(real_videos)} real videos"
    )

    fake_by_source = defaultdict(lambda: defaultdict(list))
    for video_path, video_info in fake_videos:
        source = video_info["generation_source_label"]
        semantic = video_info["semantic_source_label"]
        fake_by_source[source][semantic].append((video_path, video_info))

    sampled_fake_videos = []
    selected_video_paths = set()
    for _, real_video_info in sampled_real_videos:
        real_semantic = real_video_info["semantic_source_label"]
        for _, semantic_groups in fake_by_source.items():
            if real_semantic in semantic_groups:
                available_videos = [
                    v for v in semantic_groups[real_semantic]
                    if v[0] not in selected_video_paths
                ]
                if available_videos:
                    selected_video = random.choice(available_videos)
                    sampled_fake_videos.append(selected_video)
                    selected_video_paths.add(selected_video[0])

    result = sampled_real_videos + sampled_fake_videos

    fake_by_source_result = defaultdict(int)
    for _, video_info in sampled_fake_videos:
        source = video_info["generation_source_label"]
        fake_by_source_result[source] += 1

    source_info_parts = []
    for source, sampled_count in fake_by_source_result.items():
        total_count = sum(
            1 for _, v_info in fake_videos
            if v_info["generation_source_label"] == source
        )
        ratio = sampled_count / total_count if total_count > 0 else 0
        source_info_parts.append(
            f"{source}: {sampled_count}/{total_count}({ratio:.1%})"
        )

    logging.info(
        "Semantic-aware sampling - Real: %d, Fake: %d, Sources: %s",
        len(sampled_real_videos),
        len(sampled_fake_videos),
        ", ".join(source_info_parts),
    )

    fake_video_paths = [v[0] for v in sampled_fake_videos]
    unique_video_paths = set(fake_video_paths)
    if len(fake_video_paths) != len(unique_video_paths):
        logging.warning(
            f"Found {len(fake_video_paths) - len(unique_video_paths)} "
            f"duplicate fake videos"
        )

    return result


def _random_sampling(
    real_videos: List[Tuple[str, Dict[str, Any]]],
    fake_videos: List[Tuple[str, Dict[str, Any]]],
    sample_ratio: float,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Random sampling: Randomly sample specified ratio from real videos, and
    randomly sample fake videos from each generation source with the same
    ratio.

    Args:
        real_videos (List[Tuple[str, Dict[str, Any]]]): List of real video
            metadata tuples.
        fake_videos (List[Tuple[str, Dict[str, Any]]]): List of fake (ai-generated)
            video metadata tuples.
        sample_ratio (float): Sampling ratio (0.0-1.0).

    Returns:
        List[Tuple[str, Dict[str, Any]]]: Sampled video metadata list.
    """
    num_real_to_sample = max(1, int(len(real_videos) * sample_ratio))
    sampled_real_videos = random.sample(
        real_videos, min(num_real_to_sample, len(real_videos))
    )

    logging.info(
        f"Random sampling - Sampling {len(sampled_real_videos)} from "
        f"{len(real_videos)} real videos"
    )

    fake_by_source = defaultdict(list)
    for video_path, video_info in fake_videos:
        source = video_info["generation_source_label"]
        fake_by_source[source].append((video_path, video_info))

    sampled_fake_videos = []
    source_statistics = {}
    for source_type, source_videos in fake_by_source.items():
        num_source_to_sample = max(1, int(len(source_videos) * sample_ratio))
        sampled_from_source = random.sample(
            source_videos, min(num_source_to_sample, len(source_videos))
        )
        sampled_fake_videos.extend(sampled_from_source)
        source_statistics[source_type] = {
            "total": len(source_videos),
            "sampled": len(sampled_from_source),
            "ratio": len(sampled_from_source) / len(source_videos),
        }

    result = sampled_real_videos + sampled_fake_videos

    source_info_parts = []
    for source, stats in source_statistics.items():
        sampled_count = stats["sampled"]
        total_count = stats["total"]
        ratio = stats["ratio"]
        source_info_parts.append(
            f"{source}: {sampled_count}/{total_count}({ratio:.1%})"
        )

    logging.info(
        "Random sampling - Real: %d, Fake: %d, Sources: %s",
        len(sampled_real_videos),
        len(sampled_fake_videos),
        ", ".join(source_info_parts),
    )

    return result
