import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# function for analysing the training histories
def create_history_plots(
    history,
    model_name,
    img_size,
    learning_rate,
    batch_size,
    augmentation_strength,
    normalisation_scheme,
    path="",
):
    """
    Creates and saves plots for training and validation loss and accuracy.

    Args:
        history (dict): Dictionary containing training history.
        model_name (str): Name of the model.
        img_size (tuple): Image size used.
        learning_rate (float): Learning rate used.
        batch_size (int): Batch size used.
        augmentation_strength (int): Augmentation strength used.
        normalisation_scheme (str): Normalisation scheme used.
        path (str, optional): Directory to save the plots. Defaults to "".
    """

    txt = f"Model: {model_name}; Image Size: {img_size}; Learning Rate: {learning_rate}; Batch Size: {batch_size}; Augmentation Strength: {augmentation_strength}; Normalisation Scheme: {normalisation_scheme}."

    # plot loss
    plt.figure(figsize=(8, 5))
    plt.subplots_adjust(bottom=0.2)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Model Loss per Epoch")
    plt.figtext(0.5, 0.02, txt, wrap=True, horizontalalignment="center", fontsize=10)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(path, "history_loss.png"))
    plt.show()

    # plot accuracy
    plt.figure(figsize=(8, 5))
    plt.subplots_adjust(bottom=0.2)
    plt.plot(history["train_acc"], label=f"Train Accuracy")
    plt.plot(history["val_acc"], label=f"Validation Accuracy")
    plt.title(f"Model Accurarcy per Epoch")
    plt.figtext(0.5, 0.02, txt, wrap=True, horizontalalignment="center", fontsize=10)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(path, f"history_accuracy.png"))
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """
    Plots and saves the confusion matrix.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        classes (list): List of class names.
        save_path (str): Path to save the plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 5))
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
