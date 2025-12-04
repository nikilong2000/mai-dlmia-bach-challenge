import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec


def create_summary_figure(metric_type, output_filename):
    """
    Creates a summary figure for accuracy or loss.
    metric_type: 'acc' or 'loss'
    output_filename: filename to save the figure
    """

    # Configuration
    figures_dir = "figures"

    # Row definitions (Model Alias, Display Name)
    models = [
        ("resnet", "ResNet101\n(ImageNet)"),
        ("densenet1", "DenseNet161\n(ImageNet)"),
        ("densenet2", "DenseNet161\n(BACH)"),
    ]

    # Column definitions (Augm Suffix, Display Name)
    augmentations = [
        ("augm0", "Augmentation 0\n(None)"),
        ("augm1", "Augmentation 1\n(Simple Geometric)"),
        ("augm2", "Augmentation 2\n(Complex Geometric)"),
        ("augm3", "Augmentation 3\n(Photometric)"),
    ]

    # Create figure
    # 4 rows (1 header + 3 models), 5 columns (1 header + 4 augmentations)
    fig = plt.figure(figsize=(20, 12))

    # Use GridSpec for better control
    # height_ratios: Header row can be smaller? No, let's keep equal for simplicity or adjust.
    # width_ratios: Header col can be smaller?
    gs = GridSpec(
        4, 5, figure=fig, width_ratios=[0.5, 1, 1, 1, 1], height_ratios=[0.2, 1, 1, 1]
    )

    # --- Header Row (Augmentation Labels) ---
    # Cell (0,0) - Top Left Corner
    ax_corner = fig.add_subplot(gs[0, 0])
    ax_corner.axis("off")
    # ax_corner.text(0.5, 0.5, "Model \\\nAugmentation", ha='center', va='center', fontsize=12, fontweight='bold')

    for col_idx, (augm_suffix, augm_name) in enumerate(augmentations):
        ax = fig.add_subplot(gs[0, col_idx + 1])
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            augm_name,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )

    # --- Content Rows ---
    for row_idx, (model_alias, model_name) in enumerate(models):
        grid_row = row_idx + 1

        # Header Column (Model Labels)
        ax_header = fig.add_subplot(gs[grid_row, 0])
        ax_header.axis("off")
        ax_header.text(
            0.5,
            0.5,
            model_name,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )

        # Data Columns (Plots)
        for col_idx, (augm_suffix, _) in enumerate(augmentations):
            grid_col = col_idx + 1

            # Construct filename
            # e.g. resnet_augm0_acc.png
            filename = f"{model_alias}_{augm_suffix}_{metric_type}.png"
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

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, output_filename), dpi=300)
    print(f"Saved {output_filename} to {figures_dir}")
    plt.close()


if __name__ == "__main__":
    # Create Accuracy Figure
    create_summary_figure("acc", "summary_accuracy_grid.png")

    # Create Loss Figure
    create_summary_figure("loss", "summary_loss_grid.png")
