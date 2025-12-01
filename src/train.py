import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from src.dataset import BachDataset, get_stratified_split
from src.model import ResNet18Model, ResNet101Model, DenseNet161Model
from src.utils import load_config
from src.visualisations import create_history_plots


def train(model_name="resnet18", save_path=None, normalisation_scheme="imagenet"):
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

    # if running without ensemble
    if save_path is None:
        BEST_MODEL_PATH = config["paths"]["best_model_path"]
    else:
        BEST_MODEL_PATH = save_path

    HISTORY_DIR = config["paths"]["history_dir"]

    print(f"Using device: {DEVICE}")
    print(f"Training Model: {model_name}")
    print(f"Normalisation Scheme: {normalisation_scheme}")

    # baseline transformations (no augmentation)
    baseline_transform = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=MEAN,
                std=STD,
            ),
        ]
    )

    # laoding the dataset via custom class
    dataset = BachDataset(root_dir=DATA_DIR, transform=baseline_transform)

    # stratified split to ensure equal class distribution
    train_dataset, val_dataset, _ = get_stratified_split(dataset, SPLIT)

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
    if model_name == "resnet18":
        model = ResNet18Model(num_classes=NUM_CLASSES).to(DEVICE)
    elif model_name == "resnet101":
        model = ResNet101Model(num_classes=NUM_CLASSES).to(DEVICE)
    elif "densenet161" in model_name:  # matches densenet161_1, densenet161_2 etc
        model = DenseNet161Model(num_classes=NUM_CLASSES).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    individual_run_path = f"_{model_name}_bs{BATCH_SIZE}_lr{LEARNING_RATE}"

    if not os.path.exists(individual_run_path):
        os.makedirs(individual_run_path)

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

    # evaluation
    print("\n--- Creating history plots ---")

    if not os.path.exists(os.path.join(individual_run_path, HISTORY_DIR)):
        os.makedirs(os.path.join(individual_run_path, HISTORY_DIR))

    create_history_plots(
        history,
        path=os.path.join(individual_run_path, HISTORY_DIR),
    )


def execute_training(
    model, train_loader, val_loader, criterion, optimiser, num_epochs, device, save_path
):
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
