import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# function for analysing the training histories
def create_history_plots(
    history,
    model_name,
    learning_rate,
    batch_size,
    augmentation_strength,
    normalisation_scheme,
    path="",
):

    txt = f"Model: {model_name}; Learning Rate: {learning_rate}; Batch Size: {batch_size}; Augmentation Strength: {augmentation_strength}; Normalisation Scheme: {normalisation_scheme}."

    # plot loss
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Model Loss per Epoch")
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment="center", fontsize=12)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(path, "history_loss.png"))
    plt.show()

    # plot accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_acc"], label=f"Train Accuracy")
    plt.plot(history["val_acc"], label=f"Validation Accuracy")
    plt.title(f"Model Accurarcy per Epoch")
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment="center", fontsize=12)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(path, f"history_accuracy.png"))
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """
    Plots a confusion matrix using seaborn.

    Args:
        y_true (list or array): True labels.
        y_pred (list or array): Predicted labels.
        classes (list): List of class names.
        save_path (str): Path to save the plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.show()
