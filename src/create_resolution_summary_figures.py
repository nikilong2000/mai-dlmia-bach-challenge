import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec


def create_resolution_summary_figure(metric_type, output_filename):
    """
    Creates a summary figure for accuracy or loss comparing resolutions.
    metric_type: 'acc' or 'loss'
    output_filename: filename to save the figure
    """

    # Configuration
    output_dir = "figures"

    # Source directories
    dir_256 = os.path.join("figures", "run1_augmentations")
    dir_512 = os.path.join("figures", "run2_resolution")

    # Row definitions (Model Alias, Display Name)
    models = [
        ("resnet", "ResNet101\n(ImageNet)"),
        ("densenet1", "DenseNet161\n(ImageNet)"),
        ("densenet2", "DenseNet161\n(BACH)"),
    ]

    # Column definitions (Resolution Label, Directory, Filename Pattern)
    # Pattern is a function that takes (alias, metric) and returns filename
    resolutions = [
        (
            "256x256 (BS=32)\nAugm Strength 3",
            dir_256,
            lambda alias, metric: f"{alias}_augm3_{metric}.png",
        ),
        (
            "512x512 (BS=8)\nAugm Strength 3",
            dir_512,
            lambda alias, metric: f"{alias}_512_{metric}.png",
        ),
    ]

    # Create figure
    # 4 rows (1 header + 3 models), 3 columns (1 header + 2 resolutions)
    fig = plt.figure(figsize=(12, 12))

    gs = GridSpec(
        4, 3, figure=fig, width_ratios=[0.5, 1, 1], height_ratios=[0.2, 1, 1, 1]
    )

    # --- Header Row (Resolution Labels) ---
    # Cell (0,0) - Top Left Corner
    ax_corner = fig.add_subplot(gs[0, 0])
    ax_corner.axis("off")

    for col_idx, (res_name, _, _) in enumerate(resolutions):
        ax = fig.add_subplot(gs[0, col_idx + 1])
        ax.axis("off")
        ax.text(
            0.5, 0.5, res_name, ha="center", va="center", fontsize=14, fontweight="bold"
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
        for col_idx, (_, source_dir, pattern_func) in enumerate(resolutions):
            grid_col = col_idx + 1

            filename = pattern_func(model_alias, metric_type)
            filepath = os.path.join(source_dir, filename)

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
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_filename} to {output_dir}")
    plt.close()


if __name__ == "__main__":
    # Create Accuracy Figure
    create_resolution_summary_figure("acc", "summary_resolution_accuracy_grid.png")

    # Create Loss Figure
    create_resolution_summary_figure("loss", "summary_resolution_loss_grid.png")
