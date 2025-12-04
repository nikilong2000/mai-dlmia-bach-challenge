import os
import torch
import argparse
import sys
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report

# Add project root to path if running as script
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils import load_config
from src.dataset import BachDataset, get_holdout_split
from src.model import ResNet18Model, ResNet101Model, DenseNet121Model, DenseNet161Model
from src.augmentations import get_transform


def evaluate_single_fold(model_name, fold, normalisation_scheme, use_test_set=True):
    """
    Evaluates a specific model fold on the Hold-out Test Set.

    Args:
        model_name (str): Name of the model architecture (e.g., 'resnet101').
        fold (int): The fold number (0-4).
        normalisation_scheme (str): 'imagenet' or 'bach'.
        use_test_set (bool): If True, evaluates on the 10% hold-out test set.
                             If False, could be extended to evaluate on the fold's validation set (not implemented here for simplicity).
    """
    config = load_config()

    # Configuration
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

    # 1. Load Data
    # We use augmentation_strength=0 for validation/testing
    transform = get_transform(
        augmentation_strength=0, normalisation_scheme=normalisation_scheme
    )

    dataset = BachDataset(root_dir=DATA_DIR, transform=transform)

    if use_test_set:
        # Get the hold-out test set indices
        _, test_indices = get_holdout_split(dataset, test_size=0.1)
        eval_dataset = Subset(dataset, test_indices)
        print(f"Evaluating on Hold-out Test Set ({len(eval_dataset)} samples)")
    else:
        # Placeholder for validation set logic if needed
        print(
            "Evaluation on specific fold validation set is not fully implemented in this script."
        )
        return

    eval_loader = DataLoader(
        eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # 2. Load Model
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

    # Construct path to weights
    # Path format: results/{model_name}_bs{batch_size}_lr{learning_rate}/fold_{fold}/best_model.pth
    # Note: The folder name might vary if you changed config.
    # Assuming standard naming convention from train.py
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

    # 3. Evaluation Loop
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

    args = parser.parse_args()

    evaluate_single_fold(args.model, args.fold, args.norm)
