import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import classification_report


from src.dataset import BachDataset, get_stratified_split
from src.model import ResNet18Model
from src.utils import load_config


def evaluate():
    """
    Evaluates the trained model on the test dataset.
    Loads the best model weights, computes accuracy, and prints a classification report.
    """
    config = load_config()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    DATA_DIR = os.path.join(project_root, config["paths"]["img_dir"])
    BATCH_SIZE = config["hyperparameters"]["batch_size"]
    MEAN = config["data"]["normalisation"]["mean"]
    STD = config["data"]["normalisation"]["std"]
    NUM_CLASSES = len(config["data"]["classes"])
    SPLIT = config["data"]["split"]
    IMG_SIZE = tuple(config["hyperparameters"]["img_size"])
    MODEL_SAVE_PATH = config["paths"]["best_model_path"]
    DEVICE = torch.device("mps" if torch.mps.is_available() else "cpu")
    NUM_WORKERS = config["execution"]["num_workers"]

    print(f"Using device: {DEVICE}")

    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=MEAN,
                std=STD,
            ),
        ]
    )

    print(f"Loading dataset from: {DATA_DIR}")
    dataset = BachDataset(root_dir=DATA_DIR, transform=evaluation_transform)

    # splitting the dataset
    _, _, test_dataset = get_stratified_split(dataset, SPLIT)

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )

    print(f"Number of Evaluation Samples: {len(test_dataset)}")

    # initialising the model architecture
    model = ResNet18Model(num_classes=NUM_CLASSES).to(DEVICE)

    # loading the trained model weights
    print(f"Loading model weights from {MODEL_SAVE_PATH}")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))

    # setting the model to evaluation mode
    model.eval()

    correct = 0
    total = 0

    # lists to store all predictions and labels for potential further analysis
    all_preds = []
    all_labels = []

    # disabling gradient calculation for inference
    with torch.no_grad():
        loop = tqdm(test_loader, desc="Evaluating")
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # forward pass
            outputs = model(images)

            # get predictions
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # store predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # update the progress bar with the current accuracy
            loop.set_postfix(acc=100 * correct / total)

    # calculating and printing the overall accuracy
    accuracy = 100 * correct / total
    print(f"Overall Accuracy: {accuracy:.2f}%")

    # print a detailed classification report using scikit-learn
    class_names = config["data"]["classes"]
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
