import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec


def create_cm_summary_figure(output_filename):
    """
    Creates a summary figure for confusion matrices of all 15 models.
    output_filename: filename to save the figure
    """

    # Configuration
    figures_dir = os.path.join("figures", "grand_ensemble_confusion_matrices")
    output_dir = "figures"

    # Row definitions (Model Alias, Display Name)
    models = [
        ("resnet101", "ResNet101\n(ImageNet)"),
        ("densenet161_1", "DenseNet161\n(ImageNet)"),
        ("densenet161_2", "DenseNet161\n(BACH)"),
    ]

    # Column definitions (Fold)
    folds = [0, 1, 2, 3, 4]

    # Create figure
    # 4 rows (1 header + 3 models), 6 columns (1 header + 5 folds)
    fig = plt.figure(figsize=(25, 15))

    # Reduced spacing
    gs = GridSpec(
        4,
        6,
        figure=fig,
        width_ratios=[0.15, 1, 1, 1, 1, 1],
        height_ratios=[0.08, 1, 1, 1],
        wspace=0.05,
        hspace=0.05,
    )

    # --- Header Row (Fold Labels) ---
    # Cell (0,0) - Top Left Corner
    ax_corner = fig.add_subplot(gs[0, 0])
    ax_corner.axis("off")

    for col_idx, fold in enumerate(folds):
        ax = fig.add_subplot(gs[0, col_idx + 1])
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            f"Fold {fold}",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

    # --- Content Rows ---
    for row_idx, (model_name, display_name) in enumerate(models):
        grid_row = row_idx + 1

        # Header Column (Model Labels)
        ax_header = fig.add_subplot(gs[grid_row, 0])
        ax_header.axis("off")
        ax_header.text(
            0.5,
            0.5,
            display_name,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

        # Data Columns (Plots)
        for col_idx, fold in enumerate(folds):
            grid_col = col_idx + 1

            # Construct filename
            # e.g. resnet101_fold0_cm.png
            filename = f"{model_name}_fold{fold}_cm.png"
            filepath = os.path.join(figures_dir, filename)

            ax_plot = fig.add_subplot(gs[grid_row, grid_col])

            if os.path.exists(filepath):
                img = mpimg.imread(filepath)
                ax_plot.imshow(img)
                ax_plot.axis("off")
            else:
                ax_plot.axis("off")
                ax_plot.text(
                    0.5,
                    0.5,
                    f"File not found:\n{filename}",
                    ha="center",
                    va="center",
                    color="red",
                )

    # plt.tight_layout() # tight_layout might conflict with custom GridSpec spacing
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved {output_filename} to {output_dir}")
    plt.close()


def create_aggregated_cm_figure(output_filename):
    """
    Creates a figure with 3 confusion matrices, one for each model, aggregated across all folds.
    """
    figures_dir = os.path.join("figures", "grand_ensemble_confusion_matrices")
    output_dir = "figures"

    models = [
        ("resnet101", "ResNet101 (ImageNet)"),
        ("densenet161_1", "DenseNet161 (ImageNet)"),
        ("densenet161_2", "DenseNet161 (BACH)"),
    ]

    folds = range(5)
    class_names = [
        "Benign",
        "InSitu",
        "Invasive",
        "Normal",
    ]  # Hardcoded or load from config

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    for idx, (model_name, display_name) in enumerate(models):
        total_cm = None

        for fold in folds:
            npy_filename = f"{model_name}_fold{fold}_cm.npy"
            npy_path = os.path.join(figures_dir, npy_filename)

            if os.path.exists(npy_path):
                cm = np.load(npy_path)
                if total_cm is None:
                    total_cm = cm
                else:
                    total_cm += cm
            else:
                print(f"Warning: {npy_path} not found.")

        if total_cm is not None:
            sns.heatmap(
                total_cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=axes[idx],
                xticklabels=class_names,
                yticklabels=class_names,
                annot_kws={"size": 14},
            )
            axes[idx].set_title(
                f"{display_name}\n(Aggregated 5 Folds)", fontsize=16, fontweight="bold"
            )
            axes[idx].set_xlabel("Predicted", fontsize=12)
            axes[idx].set_ylabel("True", fontsize=12)
            axes[idx].tick_params(axis="both", which="major", labelsize=10)
        else:
            axes[idx].text(0.5, 0.5, "No Data", ha="center", va="center")

    plt.tight_layout()
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved {output_filename} to {output_dir}")
    plt.close()


if __name__ == "__main__":
    create_cm_summary_figure("summary_confusion_matrices_grid.png")
    create_aggregated_cm_figure("aggregated_confusion_matrices.png")
