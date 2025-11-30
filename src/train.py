import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from src.dataset import BachDataset
from src.model import ResNet18Model
from src.dataset import BachDataset
from src.utils import load_config


def train():
    config = load_config()

    # setting the global variables according to config
    BATCH_SIZE = config["hyperparameters"]["batch_size"]
    LEARNING_RATE = config["hyperparameters"]["learning_rate"]
    MEAN = config["data"]["normalisation"]["mean"]
    STD = config["data"]["normalisation"]["std"]
    NUM_CLASSES = len(config["data"]["classes"])
    NUM_EPOCHS = config["hyperparameters"]["num_epochs"]
    IMG_SIZE = tuple(config["hyperparameters"]["img_size"])
    DATA_DIR = config["data"]["img_dir"]
    DEVICE = torch.device("mps" if torch.mps.is_available() else "cpu")
    NUM_WORKERS = config["execution"]["num_workers"]
    MODEL_SAVE_PATH = config["model"]["model_weights"]

    print(f"Using device: {DEVICE}")

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

    # splitting the dataset
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

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
    model = ResNet18Model(num_classes=NUM_CLASSES).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # training loop
    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

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
                images, labels = images.to(DEVICE), labels.to(DEVICE)
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
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Saved best model to {MODEL_SAVE_PATH}")
