"""Plotting functionality"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_predictions(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    labels: pd.Series | None = None,
    show_labels: bool = False,
    streamlit: bool = True,
) -> None:
    depth = df["Depth (m)"]
    qc = df["qc (MPa)"]
    fs = df["fs (kPa)"]

    plt.rcParams.update(
        {
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 18,
        }
    )

    fig, axs = plt.subplots(
        1, 4 if labels is not None else 3, figsize=(20, 10), sharey=True
    )

    # Plot qc (MPa) vs Depth
    axs[0].plot(qc, depth, color="b")
    axs[0].set_xlabel("qc (MPa)")
    axs[0].set_ylabel("Depth (m)")
    axs[0].invert_yaxis()
    axs[0].set_title("qc (MPa) vs Depth")

    # Plot fs (kPa) vs Depth
    axs[1].plot(fs, depth, color="g")
    axs[1].set_xlabel("fs (kPa)")
    axs[1].set_title("fs (kPa) vs Depth")

    # Plot predicted labels vs Depth
    axs[2].plot(y_pred, depth, color="r")
    axs[2].set_xlabel("Predicted Classes")
    axs[2].set_title("Predicted vs Depth")

    # Plot true labels vs Depth (only if labels are provided)
    if labels is not None and show_labels:
        axs[3].plot(labels, depth, color="orange")
        axs[3].set_xlabel("True Classes")
        axs[3].set_title("True Labels vs Depth")

    plt.tight_layout()
    if streamlit:
        st.pyplot(fig)
    else:
        plt.show()


def plot_confusion_matrix(
    y_true: list[Any],
    y_pred: list[Any],
    class_mapping: dict[int, str],
    normalize: str = "true",
    add_black_lines: bool = True,
) -> plt.Figure:
    """
    Plots a confusion matrix with options for customization.

    Parameters:
    y_true (list[Any]): True labels.
    y_pred (list[Any]): Predicted labels.
    class_mapping (dict[int, str]): Mapping of class numbers to class names.
    normalize (str, optional): Normalization option for confusion matrix. Defaults to 'true'.
    add_black_lines (bool, optional): Whether to add black grid lines around each square. Defaults to True.

    Example class_mapping:

    soil_classification = {
    1: "gravel",
    4: "sand to gravel",
    3: "coarse grained organic soils",
    5: "sand",
    2: "fine grained organic soils",
    6: "silt to fine sand",
    7: "clay to silt"
    }

    """
    # Update the labels using the mapping table with map function
    class_labels = list(class_mapping.values())
    class_label_numbers = list(class_mapping.keys())

    # Generate the confusion matrix with labels in the desired order
    conf_matrix = confusion_matrix(
        y_true, y_pred, labels=class_label_numbers, normalize=normalize
    )

    # Round the values in the confusion matrix to a maximum of 3 digits behind the comma
    conf_matrix = np.round(conf_matrix, 3)

    # Visualise the confusion matrix with an optional thin black line around each square
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_matrix, display_labels=class_labels
    )
    disp.plot(cmap="Blues", ax=ax, colorbar=False)

    # Toggle black grid lines on or off
    if add_black_lines:
        add_black_grid_lines(ax, conf_matrix)

    # Ensure labels align correctly with ticks
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_yticklabels(class_labels)

    for text in disp.text_.ravel():
        text.set_fontsize(12)  # Adjust the size to your liking

    plt.title("Confusion Matrix (recall for each class on the diagonal)")
    plt.tight_layout()
    return fig


def add_black_grid_lines(ax: plt.Axes, conf_matrix: np.ndarray) -> None:
    """
    Adds black grid lines around each square in the confusion matrix.

    Parameters:
    ax (plt.Axes): The axes object of the plot.
    conf_matrix (np.ndarray): The confusion matrix data.
    """
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_edgecolor("black")
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
    ax.set_xticks(np.arange(conf_matrix.shape[1]) + 0.5, minor=True)
    ax.set_yticks(np.arange(conf_matrix.shape[0]) + 0.5, minor=True)
    ax.tick_params(which="minor", size=0)
