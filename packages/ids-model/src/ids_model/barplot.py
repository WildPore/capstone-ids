import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Union


def plot_class_distribution(
    y: Union[pd.Series, np.ndarray],
    title: str = "Class Distribution",
    figsize: tuple = (10, 6),
    save_path: str = "class_distribution_barplot.png",
) -> dict:
    if isinstance(y, np.ndarray):
        y = pd.Series(y)

    class_counts = y.value_counts()
    total_samples = len(y)

    fig, ax = plt.subplots(figsize=figsize)

    # Both of these lines are fine, plt.cm.viridis exists...
    # And I'm not going to bother dealing with class_counts.value.
    # It is partially the right type, it has one of the types in the union
    # for that parameter. So whatever.
    colors = plt.cm.viridis(np.linspace(0, 1, len(class_counts)))  # type:ignore
    bars = ax.bar(class_counts.index.astype(str), class_counts.values, color=colors)  # type:ignore

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    plt.xticks(rotation=90)

    for bar, count in zip(bars, class_counts.values):
        height = bar.get_height()
        percentage = (count / total_samples) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{count}\n({percentage:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    imbalance_ratio = class_counts.max() / class_counts.min()

    return {
        "class_counts": class_counts.to_dict(),
        "total_samples": total_samples,
        "imbalance_ratio": imbalance_ratio,
        "percentages": {k: (v / total_samples) * 100 for k, v in class_counts.items()},
    }
