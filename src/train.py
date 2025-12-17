import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from src.dataset import (
    BachDataset,
    get_stratified_split,
    get_holdout_split,
    get_cv_folds,
)
from src.model import ResNet18Model, ResNet101Model, DenseNet121Model, DenseNet161Model
from src.utils import load_config
from src.visualisations import create_history_plots, plot_confusion_matrix
from src.augmentations import get_transform


def train(
    model_name="resnet18",
    learning_rate=0.0001,
    augmentation_strength=0,
    normalisation_scheme="imagenet",
):
    """
    Trains a single model with the specified configuration.

    Args:
        model_name (str, optional): Name of the model architecture. Defaults to "resnet18".
        learning_rate (float, optional): Learning rate for the optimiser. Defaults to 0.0001.
        augmentation_strength (int, optional): Strength of data augmentation. Defaults to 0.
        normalisation_scheme (str, optional): Normalisation scheme to use. Defaults to "imagenet".
    """
    config = load_config()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # setting the global variables according to config
    DATA_DIR = os.path.join(project_root, config["paths"]["img_dir"])
    BATCH_SIZE = config["hyperparameters"]["batch_size"]
    LEARNING_RATE = config["hyperparameters"]["learning_rate"]
    MEAN = config["data"]["normalisation"][normalisation_scheme]["mean"]
    STD = config["data"]["normalisation"][normalisation_scheme]["std"]
    NUM_CLASSES = len(config["data"]["classes"])
    SPLIT = config["data"]["split"]
    NUM_EPOCHS = config["hyperparameters"]["num_epochs"]
    IMG_SIZE = tuple(config["hyperparameters"]["img_size"])
    DEVICE = torch.device("mps" if torch.mps.is_available() else "cpu")
    NUM_WORKERS = config["execution"]["num_workers"]
    BEST_MODEL_PATH = config["paths"]["best_model_path"]
    HISTORY_DIR = config["paths"]["history_dir"]

    print("\n--- Initialising Training Configuration ---")
    print(f"Using device: {DEVICE}")
    print(f"Training Model: {model_name}")
    print(f"Normalisation Scheme: {normalisation_scheme}")

    # get augmentation transforms
    print("\n--- Loading Dataset and Transforms ---")
    train_transform = get_transform(augmentation_strength, normalisation_scheme)
    val_transform = get_transform(0, normalisation_scheme)

    # laoding the dataset via custom class
    train_dataset_full = BachDataset(root_dir=DATA_DIR, transform=train_transform)
    val_dataset_full = BachDataset(root_dir=DATA_DIR, transform=val_transform)

    # stratified split to ensure equal class distribution
    train_dataset, val_dataset, _ = get_stratified_split(train_dataset_full, SPLIT)

    # apply validation transform to validation dataset
    val_dataset = Subset(val_dataset_full, val_dataset.indices)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    print(
        f"Number of Training Samples: {len(train_dataset)}, Number of Validation Samples: {len(val_dataset)}"
    )

    # setting up the model
    print("\n--- Initialising Model ---")
    if model_name == "resnet18":
        model = ResNet18Model(num_classes=NUM_CLASSES).to(DEVICE)
    elif model_name == "resnet101":
        model = ResNet101Model(num_classes=NUM_CLASSES).to(DEVICE)
    elif "densenet121" in model_name:
        model = DenseNet121Model(num_classes=NUM_CLASSES).to(DEVICE)
    elif "densenet161" in model_name:
        model = DenseNet161Model(num_classes=NUM_CLASSES).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)

    individual_run_path = os.path.join(
        "results",
        f"{model_name}_res{IMG_SIZE[0]}_bs{BATCH_SIZE}_lr{learning_rate}_augm{augmentation_strength}_norm{normalisation_scheme}",
    )

    if not os.path.exists(individual_run_path):
        os.makedirs(individual_run_path)

    print("\n--- Starting Training Loop ---")
    history = execute_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimiser=optimiser,
        num_epochs=NUM_EPOCHS,
        device=DEVICE,
        save_path=os.path.join(individual_run_path, BEST_MODEL_PATH),
    )

    # save metrics to file
    print("\n--- Saving Metrics ---")
    metrics_path = os.path.join(individual_run_path, "metrics.txt")
    with open(metrics_path, "w") as f:
        for key, value in history.items():
            f.write(f"{key}: {value}\n")
    print(f"Saved metrics to {metrics_path}")

    # evaluation
    print("\n--- Creating history plots ---")

    if not os.path.exists(os.path.join(individual_run_path, HISTORY_DIR)):
        os.makedirs(os.path.join(individual_run_path, HISTORY_DIR))

    create_history_plots(
        history,
        model_name,
        IMG_SIZE,
        LEARNING_RATE,
        BATCH_SIZE,
        augmentation_strength,
        normalisation_scheme,
        path=os.path.join(individual_run_path, HISTORY_DIR),
    )

    # confusion matrix
    print("\n--- Creating confusion matrix ---")
    model.load_state_dict(
        torch.load(os.path.join(individual_run_path, BEST_MODEL_PATH))
    )
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    plot_confusion_matrix(
        all_labels,
        all_preds,
        classes=config["data"]["classes"],
        save_path=os.path.join(
            individual_run_path, HISTORY_DIR, "confusion_matrix.png"
        ),
    )


def train_k_fold(
    model_name="resnet18",
    learning_rate=0.0001,
    augmentation_strength=0,
    normalisation_scheme="imagenet",
):
    """
    Trains a model using K-Fold cross-validation.

    Args:
        model_name (str, optional): Name of the model architecture. Defaults to "resnet18".
        learning_rate (float, optional): Learning rate for the optimiser. Defaults to 0.0001.
        augmentation_strength (int, optional): Strength of data augmentation. Defaults to 0.
        normalisation_scheme (str, optional): Normalisation scheme to use. Defaults to "imagenet".
    """
    config = load_config()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # setting the global variables according to config
    DATA_DIR = os.path.join(project_root, config["paths"]["img_dir"])
    BATCH_SIZE = config["hyperparameters"]["batch_size"]
    LEARNING_RATE = config["hyperparameters"]["learning_rate"]
    NUM_CLASSES = len(config["data"]["classes"])
    NUM_EPOCHS = config["hyperparameters"]["num_epochs"]
    IMG_SIZE = tuple(config["hyperparameters"]["img_size"])
    DEVICE = torch.device("mps" if torch.mps.is_available() else "cpu")
    NUM_WORKERS = config["execution"]["num_workers"]
    BEST_MODEL_PATH = config["paths"]["best_model_path"]
    HISTORY_DIR = config["paths"]["history_dir"]
    K_FOLDS = config["hyperparameters"]["k_folds"]

    print("\n--- Initialising K-Fold Training Configuration ---")
    print(f"Using device: {DEVICE}")
    print(f"Training Model: {model_name}")
    print(f"Normalisation Scheme: {normalisation_scheme}")
    print(f"K-Folds: {K_FOLDS}")

    # get augmentation transforms
    print("\n--- Loading Dataset and Transforms ---")
    train_transform = get_transform(augmentation_strength, normalisation_scheme)
    val_transform = get_transform(0, normalisation_scheme)

    # loading the dataset via custom class
    dataset_train_aug = BachDataset(root_dir=DATA_DIR, transform=train_transform)
    dataset_val_aug = BachDataset(root_dir=DATA_DIR, transform=val_transform)

    # split into dev and hold-out test
    dev_indices, test_indices = get_holdout_split(dataset_train_aug, test_size=0.1)
    print(
        f"Total Dev Samples: {len(dev_indices)}, Hold-out Test Samples: {len(test_indices)}"
    )

    # generate k-folds from dev set
    folds = get_cv_folds(dataset_train_aug, dev_indices, k_folds=K_FOLDS)

    base_run_path = os.path.join(
        "results",
        f"{model_name}_bs{BATCH_SIZE}_lr{learning_rate}",
    )

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\n--- Starting Fold {fold_idx+1}/{K_FOLDS} ---")

        # create Subsets
        # train subset uses dataset with train augmentations
        train_subset = Subset(dataset_train_aug, train_idx)
        # val subset uses dataset with val augmentations
        val_subset = Subset(dataset_val_aug, val_idx)

        train_loader = DataLoader(
            train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
        )
        val_loader = DataLoader(
            val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
        )

        print(
            f"Fold {fold_idx+1} - Train samples: {len(train_subset)}, Val samples: {len(val_subset)}"
        )

        # setting up the model (fresh for each fold)
        if model_name == "resnet18":
            model = ResNet18Model(num_classes=NUM_CLASSES).to(DEVICE)
        elif model_name == "resnet101":
            model = ResNet101Model(num_classes=NUM_CLASSES).to(DEVICE)
        elif "densenet121" in model_name:
            model = DenseNet121Model(num_classes=NUM_CLASSES).to(DEVICE)
        elif "densenet161" in model_name:
            model = DenseNet161Model(num_classes=NUM_CLASSES).to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        optimiser = optim.Adam(model.parameters(), lr=learning_rate)

        fold_run_path = os.path.join(base_run_path, f"fold_{fold_idx}")
        if not os.path.exists(fold_run_path):
            os.makedirs(fold_run_path)

        history = execute_training(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimiser=optimiser,
            num_epochs=NUM_EPOCHS,
            device=DEVICE,
            save_path=os.path.join(fold_run_path, BEST_MODEL_PATH),
        )

        # save metrics to file
        metrics_path = os.path.join(fold_run_path, "metrics.txt")
        with open(metrics_path, "w") as f:
            for key, value in history.items():
                f.write(f"{key}: {value}\n")

        # create history plots for this fold
        if not os.path.exists(os.path.join(fold_run_path, HISTORY_DIR)):
            os.makedirs(os.path.join(fold_run_path, HISTORY_DIR))

        create_history_plots(
            history,
            f"{model_name}_fold{fold_idx}",
            IMG_SIZE,
            LEARNING_RATE,
            BATCH_SIZE,
            augmentation_strength,
            normalisation_scheme,
            path=os.path.join(fold_run_path, HISTORY_DIR),
        )


def execute_training(
    model, train_loader, val_loader, criterion, optimiser, num_epochs, device, save_path
):
    """
    Executes the training loop for a given number of epochs.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        criterion (nn.Module): Loss function.
        optimiser (torch.optim.Optimizer): Optimiser.
        num_epochs (int): Number of epochs to train.
        device (torch.device): Device to run training on.
        save_path (str): Path to save the best model weights.

    Returns:
        dict: A dictionary containing training history (loss and accuracy).
    """
    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimiser.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loop.set_postfix(
                loss=running_loss / len(train_loader), acc=100 * correct / total
            )

        train_acc = 100 * correct / total

        # validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total

        history["train_loss"].append(running_loss / len(train_loader))
        history["val_loss"].append(val_running_loss / len(val_loader))
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch+1}: Train Accuracy: {train_acc:.2f}%, Validation Accuracy: {val_acc:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

    return history
