import os
import torch
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report
from src.utils import load_config
from src.dataset import BachDataset, get_holdout_split
from src.model import ResNet18Model, ResNet101Model, DenseNet121Model, DenseNet161Model
from src.augmentations import get_transform

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)


def evaluate_single_fold(
    model_name, fold, normalisation_scheme, use_test_set=True, save_cm=False
):
    """
    Evaluates a specific model fold on the Hold-out Test Set.

    Args:
        model_name (str): Name of the model architecture (e.g., 'resnet101').
        fold (int): The fold number (0-4).
        normalisation_scheme (str): 'imagenet' or 'bach'.
        use_test_set (bool): If True, evaluates on the 10% hold-out test set.
                             If False, could be extended to evaluate on the fold's validation set (not implemented here for simplicity).
        save_cm (bool): If True, saves the confusion matrix to figures/grand_ensemble_confusion_matrices.
    """
    config = load_config()

    # configuration
    DATA_DIR = os.path.join(project_root, config["paths"]["img_dir"])
    BATCH_SIZE = config["hyperparameters"]["batch_size"]
    LEARNING_RATE = config["hyperparameters"]["learning_rate"]
    NUM_CLASSES = len(config["data"]["classes"])
    DEVICE = torch.device("mps" if torch.mps.is_available() else "cpu")
    NUM_WORKERS = config["execution"]["num_workers"]
    BEST_MODEL_PATH = config["paths"]["best_model_path"]

    print(f"\n--- Evaluating {model_name} (Fold {fold}) ---")
    print(f"Normalisation: {normalisation_scheme}")
    print(f"Device: {DEVICE}")

    # load Data
    transform = get_transform(
        augmentation_strength=0, normalisation_scheme=normalisation_scheme
    )

    dataset = BachDataset(root_dir=DATA_DIR, transform=transform)

    if use_test_set:
        # get the hold-out test set indices
        _, test_indices = get_holdout_split(dataset, test_size=0.1)
        eval_dataset = Subset(dataset, test_indices)
        print(f"Evaluating on Hold-out Test Set ({len(eval_dataset)} samples)")
    else:
        return

    eval_loader = DataLoader(
        eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # load model
    if model_name == "resnet18":
        model = ResNet18Model(num_classes=NUM_CLASSES)
    elif model_name == "resnet101":
        model = ResNet101Model(num_classes=NUM_CLASSES)
    elif "densenet121" in model_name:
        model = DenseNet121Model(num_classes=NUM_CLASSES)
    elif "densenet161" in model_name:
        model = DenseNet161Model(num_classes=NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # construct path to weights
    run_folder = f"{model_name}_bs{BATCH_SIZE}_lr{LEARNING_RATE}"
    weights_path = os.path.join(
        project_root, "results", run_folder, f"fold_{fold}", BEST_MODEL_PATH
    )

    if not os.path.exists(weights_path):
        print(f"Error: Model weights not found at {weights_path}")
        return

    print(f"Loading weights from: {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # evaluation Loop
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in eval_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")

    class_names = config["data"]["classes"]
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    if save_cm:
        cm_dir = os.path.join(
            project_root, "figures", "grand_ensemble_confusion_matrices"
        )
        if not os.path.exists(cm_dir):
            os.makedirs(cm_dir)

        cm = confusion_matrix(all_labels, all_preds)

        # save cm data for aggregation
        np.save(os.path.join(cm_dir, f"{model_name}_fold{fold}_cm.npy"), cm)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix: {model_name} (Fold {fold})")

        filename = f"{model_name}_fold{fold}_cm.png"
        plt.savefig(os.path.join(cm_dir, filename), bbox_inches="tight", pad_inches=0.1)
        plt.close()
        print(f"Saved confusion matrix to {os.path.join(cm_dir, filename)}")

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a specific model fold.")
    parser.add_argument(
        "--model", type=str, required=True, help="Model name (e.g., resnet101)"
    )
    parser.add_argument("--fold", type=int, required=True, help="Fold number (0-4)")
    parser.add_argument(
        "--norm",
        type=str,
        default="imagenet",
        help="Normalisation scheme (imagenet/bach)",
    )
    parser.add_argument("--save_cm", action="store_true", help="Save confusion matrix")

    args = parser.parse_args()

    evaluate_single_fold(args.model, args.fold, args.norm, save_cm=args.save_cm)
