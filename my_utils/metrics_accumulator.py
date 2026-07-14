"""
Factory for a lightweight metrics accumulator used in train/eval loops.

Provides add, get_mean, get_concat, get_concat_tensor, and exists on the returned object.
"""
import numpy as np
import torch
from typing import Any


def metrics_accumulator():
    """
    Create a metrics accumulator for tracking metrics during training/evaluation.

    Returns a dictionary-like object that manages multiple metric lists and
    provides a unified interface for adding batch results and computing final
    metrics.

    The accumulator supports the following operations:
    - add(key, value): Add a value to the specified metric list.
    - get_mean(key, default=0.0): Compute mean of a metric list.
    - get_concat(key): Concatenate all numpy arrays in a metric list.
    - get_concat_tensor(key): Concatenate all torch.Tensor arrays in a metric list.
    - exists(key): Check if a metric key exists and has data.

    Returns:
        MetricsAccumulator: An accumulator object for tracking metrics.

    Examples:
        >>> metrics = metrics_accumulator()
        >>>
        >>> # Add batch results during training
        >>> for batch in dataloader:
        ...     loss = compute_loss(batch)
        ...     metrics.add("loss", loss.item())
        ...     metrics.add("base_labels", batch_labels.numpy())
        >>>
        >>> # Compute final metrics
        >>> avg_loss = metrics.get_mean("loss")
        >>> all_labels = metrics.get_concat("base_labels")
        >>>
        >>> # Check if metric exists
        >>> if metrics.exists("loss"):
        ...     print(f"Average loss: {metrics.get_mean('loss'):.4f}")
    """

    class MetricsAccumulator:
        """Accumulator for tracking metrics during training/evaluation."""

        def __init__(self):
            self.data = {}

        def add(self, key: str, value: Any):
            """
            Add a value to the specified metric list.

            Args:
                key (str): Metric name.
                value (Any): Value to add (scalar, numpy array, etc.).
            """
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)

        def get_mean(self, key: str, default: float = 0.0) -> float:
            """
            Compute mean of a metric list.

            Args:
                key (str): Metric name.
                default (float): Default value if key doesn't exist or is empty.

            Returns:
                float: Mean value of the metric.
            """
            if key not in self.data or len(self.data[key]) == 0:
                return default
            return np.mean(self.data[key])

        def get_concat(self, key: str) -> np.ndarray:
            """
            Concatenate all arrays in a metric list.

            Args:
                key (str): Metric name.

            Returns:
                np.ndarray: Concatenated array. Returns empty array if key
                    doesn't exist or is empty.
            """
            if key not in self.data or len(self.data[key]) == 0:
                return np.array([])
            return np.concatenate(self.data[key])

        def get_concat_tensor(self, key: str) -> torch.Tensor:
            """
            Concatenate all torch.Tensor arrays in a metric list.

            Args:
                key (str): Metric name.

            Returns:
                torch.Tensor: Concatenated tensor. Returns empty tensor if key
                    doesn't exist or is empty.
            """
            if key not in self.data or len(self.data[key]) == 0:
                return torch.tensor([])
            return torch.cat(self.data[key], dim=0)

        def exists(self, key: str) -> bool:
            """
            Check if a metric key exists and has data.

            Args:
                key (str): Metric name.

            Returns:
                bool: True if key exists and has data, False otherwise.
            """
            return key in self.data and len(self.data[key]) > 0

    return MetricsAccumulator()
