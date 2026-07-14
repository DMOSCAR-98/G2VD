"""
t-SNE visualization for collected video-level features.

Expects feature matrices with shape [N, D] (N samples, D dimensions) and matching label vectors.
The standard pipeline is StandardScaler -> optional PCA (max 50D) -> t-SNE, with
silhouette score computed on the standardized/PCA feature space before t-SNE.
"""
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# Colorblind-friendly academic palette (Okabe-Ito).
_OKABE_ITO = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#735F58",  # warm gray-brown
]
_GS_SOURCE_COLORS = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#735F58",  # warm gray-brown
]

_REAL_COLOR = "#9AD9A3"  # light green
_FAKE_COLOR = "#F2A5A5"  # light red
_CF_COLOR = "#7F7F7F"  # neutral gray


def visualize_tsne(
    features: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    label_type: str = "base",
    save_path: str = "tsne_vis.png",
    title: str = "t-SNE Visualization",
    perplexity: int = 40,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 300,
    return_metrics: bool = False,
) -> Tuple[Figure, Axes] | Tuple[Figure, Axes, Dict[str, Any]]:
    """
    Visualize feature distribution using t-SNE.

    Reduces high-dimensional features to 2D space and visualizes data points.
    Default project usage colors points by base real/fake labels
    (label_type="base"; Real=light green, Fake=light red). Set
    label_type="gs" to switch to generation-source coloring with the same
    pipeline. In gs mode, real samples keep the fixed light-green color, CF
    samples keep the fixed neutral-gray color, and fake generation sources use
    _GS_SOURCE_COLORS first, then tab20/husl palettes when there are more
    source labels.

    Standard visualization pipeline: apply StandardScaler first, then PCA to at
    most 50 dimensions when needed, then t-SNE to 2 dimensions. For cases with
    fewer than 15 samples, use PCA to reduce to 2D to avoid t-SNE instability.
    Silhouette score is computed on the standardized/PCA feature space before
    t-SNE, because t-SNE itself may distort global distances.

    Args:
        features (torch.Tensor | np.ndarray): Feature matrix with shape [N, D],
            where N is the number of samples and D is the feature dimension.
            Features are standardized inside this function before dimensionality
            reduction and silhouette scoring.
        labels (torch.Tensor | np.ndarray): Label vector with shape [N,].
            Label values depend on label_type:
            - If label_type="base": 0 (Real), 1 (Fake, ai-generated)
            - If label_type="gs": -1 (Real), 0,1,2,... (Fake sources), 999 (CF)
        label_type (str, optional): Type of labels. Options:
            - "base": Base labels (0=Real, 1=Fake (ai-generated))
            - "gs": Generation source labels (-1=Real, 0,1,2,...=Fake sources,
            999=CF)
            Defaults to "base".
        save_path (str, optional): Path to save the visualization image.
            Defaults to "tsne_vis.png".
        title (str, optional): Title of the plot. Defaults to
            "t-SNE Visualization".
        perplexity (int, optional): Perplexity parameter for t-SNE.
            Defaults to 40. Will be automatically adjusted based on sample size
            and capped at 50 (t-SNE best practice).
        figsize (Tuple[int, int], optional): Figure size (width, height).
            Defaults to (10, 8).
        dpi (int, optional): Resolution for saved image. Defaults to 300.
        return_metrics (bool, optional): If True, also return a metrics dict
            containing silhouette_score, num_samples, num_classes, reduction,
            and perplexity_used. Defaults to False for backward compatibility.

    Returns:
        Tuple[Figure, Axes]: matplotlib Figure and Axes objects when
            return_metrics=False. If return_metrics=True, returns
            Tuple[Figure, Axes, Dict[str, Any]], where the metrics dict records
            the silhouette score, sample/class counts, feature reduction path,
            and actual t-SNE perplexity.

    Raises:
        ValueError: If label_type is not 'gs' or 'base'.

    Examples:
        >>> import torch
        >>> from my_utils import visualize_tsne
        >>>
        >>> features = torch.randn(100, 512)
        >>>
        >>> # Example 1: Base labels (binary classification; default)
        >>> base_labels = torch.tensor([0]*50 + [1]*50)
        >>> fig, ax = visualize_tsne(
        ...     features,
        ...     labels=base_labels,
        ...     save_path="tsne_binary.png",
        ...     title="Real vs Fake Distribution"
        ... )
        >>>
        >>> # Example 2: Generation source labels
        >>> gs_labels = torch.tensor([-1]*20 + [0]*20 + [1]*20 + [2]*20 + [999]*20)  # noqa: E501
        >>> fig, ax = visualize_tsne(
        ...     features,
        ...     labels=gs_labels,
        ...     label_type="gs",
        ...     save_path="tsne_gs.png",
        ...     title="Generation Source Distribution"
        ... )
    """
    # Type conversion: Convert PyTorch tensors to numpy arrays
    # Note: .float() ensures compatibility with BFloat16/Float16 types that
    # are not directly supported by numpy
    if isinstance(features, torch.Tensor):
        features = features.float().detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.float().detach().cpu().numpy()

    features = np.asarray(features)
    labels = np.asarray(labels).reshape(-1)
    if features.ndim != 2:
        raise ValueError(
            f"features must be a 2D matrix [N, D], got shape {features.shape}."
        )
    if len(labels) != features.shape[0]:
        raise ValueError(
            f"labels length ({len(labels)}) must match features rows "
            f"({features.shape[0]})."
        )
    N, D = features.shape

    # Validate label_type
    if label_type not in ["gs", "base"]:
        raise ValueError(
            f"Invalid label_type: {label_type}. Must be 'gs' or 'base'."
        )

    # Standardize features before PCA/t-SNE and silhouette computation.
    features = StandardScaler().fit_transform(features)
    reduction = "standardized"
    if D > 50 and N > 1:
        n_comp = min(50, N - 1, D)
        features = PCA(n_components=n_comp, random_state=42).fit_transform(features)
        reduction = f"standardized+pca{n_comp}"

    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    silhouette: Optional[float]
    if 2 <= n_classes <= N - 1:
        silhouette = float(silhouette_score(features, labels))
    else:
        silhouette = None

    # Apply t-SNE or PCA for 2D visualization
    perplexity_used: Optional[int] = None
    if N == 1:
        features_2d = np.zeros((1, 2), dtype=np.float32)
    elif N < 15:
        # Use PCA for small samples to avoid t-SNE instability
        pca_dim = min(2, N, features.shape[1])
        features_2d = PCA(n_components=pca_dim, random_state=42).fit_transform(features)
        if pca_dim == 1:
            features_2d = np.pad(features_2d, ((0, 0), (0, 1)))
    else:
        # Adjust perplexity based on sample size
        # Cap maximum perplexity at 50 (t-SNE best practice)
        max_perplexity = 50
        perplexity_used = min(perplexity, max_perplexity, max(5, (N - 1) // 3))
        tsne_init = "pca" if features.shape[1] >= 2 else "random"
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity_used,
            random_state=42,
            init=tsne_init,
            learning_rate="auto",
            verbose=0,
        )
        features_2d = tsne.fit_transform(features)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get unique labels and create palette.
    # Prefer colorblind-friendly colors for paper figures.
    source_color_map = {}
    if label_type == "gs":
        source_labels = [
            label for label in unique_labels if label not in [-1, 999]
        ]
        n_sources = len(source_labels)
        if n_sources <= len(_GS_SOURCE_COLORS):
            colors = list(_GS_SOURCE_COLORS[:n_sources])
        elif n_sources <= 20:
            colors = sns.color_palette("tab20", n_colors=n_sources).as_hex()
        else:
            colors = sns.color_palette("husl", n_colors=n_sources).as_hex()
        source_color_map = {
            source_label: colors[i] for i, source_label in enumerate(source_labels)
        }
    elif n_classes <= len(_OKABE_ITO):
        colors = list(_OKABE_ITO[:n_classes])
    elif n_classes <= 20:
        colors = sns.color_palette("tab20", n_colors=n_classes).as_hex()
    else:
        colors = sns.color_palette("husl", n_colors=n_classes).as_hex()

    # Plot data points by labels.
    for label in unique_labels:
        mask = labels == label
        # Generate label name based on label_type
        if label_type == "gs":
            # Generation source labels: -1=Real, 0,1,2,...=Fake sources, 999=CF
            if label == -1:
                label_name = "Real"
                color = _REAL_COLOR
            elif label == 999:
                label_name = "CF"
                color = _CF_COLOR
            else:
                label_name = f"Gen{int(label)}"
                color = source_color_map[label]
        else:  # label_type == "base"
            # Base labels (binary): 0=Real, 1=Fake (ai-generated)
            if label == 0:
                label_name = "Real"
                color = _REAL_COLOR
            elif label == 1:
                label_name = "Fake"
                color = _FAKE_COLOR
            else:
                raise ValueError(
                    f"Invalid base label value: {label}. "
                    "Base labels should only be 0 or 1."
                )

        ax.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[color],
            s=28,
            alpha=0.6,
            label=label_name,
            edgecolors="none",
        )

    # Customize plot appearance
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=11)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    silhouette_text = (
        f"Silhouette ({reduction}): {silhouette:.4f}"
        if silhouette is not None
        else f"Silhouette ({reduction}): N/A"
    )
    ax.text(
        0.02,
        0.02,
        silhouette_text,
        transform=ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )

    # Save figure
    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)  # Release figure to avoid memory buildup when called repeatedly

    metrics = {
        "silhouette_score": silhouette,
        "num_samples": int(N),
        "num_classes": int(n_classes),
        "reduction": reduction,
        "perplexity_used": perplexity_used,
    }
    if return_metrics:
        return fig, ax, metrics
    return fig, ax
