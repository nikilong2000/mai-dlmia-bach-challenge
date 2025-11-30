import os
import matplotlib as plt


# function for analysing the training histories
def create_history_plots(history, path=""):
    # plot loss
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Model Loss per Epoch")
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
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(path, f"history_accuracy.png"))
    plt.show()
