import torch
import os
import numpy as np
from PIL import Image
from collections import Counter
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import classification_report


from src.model import ResNet101Model, DenseNet161Model, ResNet18Model, DenseNet121Model
from src.utils import load_config
import torch.nn.functional as F
from src.dataset import BachDataset, get_stratified_split, get_holdout_split
from src.train import train
from src.visualisations import plot_confusion_matrix


class EnsembleClassifier:
    """
    A classifier that ensembles multiple models for prediction.
    """
    def __init__(self, models_config, device):
        """
        Initialises the EnsembleClassifier with a list of model configurations.

        Args:
            models_config (list): List of dictionaries containing model configuration (name, normalisation).
            device (torch.device): The device to run the models on.
        """
        self.models = []
        self.device = device
        self.config = load_config()
        self.num_classes = len(self.config["data"]["classes"])
        self.img_size = tuple(self.config["hyperparameters"]["img_size"])

        for model_config in models_config:
            model_name = model_config["name"]
            normalisation_scheme = model_config["normalisation"]

            # construct path
            batch_size = self.config["hyperparameters"]["batch_size"]
            learning_rate = self.config["hyperparameters"]["learning_rate"]
            best_model_filename = self.config["paths"]["best_model_path"]

            run_folder = f"{model_name}_bs{batch_size}_lr{learning_rate}"
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            save_path = os.path.join(
                project_root, "results", run_folder, best_model_filename
            )

            # initialise model architecture
            if model_name == "resnet18":
                model = ResNet18Model(num_classes=self.num_classes)
            elif model_name == "resnet101":
                model = ResNet101Model(num_classes=self.num_classes)
            elif "densenet161" in model_name:
                model = DenseNet161Model(num_classes=self.num_classes)

            # load weights for evaluation
            if os.path.exists(save_path):
                print(f"Loading {model_name} from {save_path}")
                model.load_state_dict(torch.load(save_path, map_location=device))
            else:
                print(
                    f"Weights for {model_name} not found at {save_path}. Cancel evaluation."
                )
                return

            model.to(device)
            model.eval()

            # create specific transform for this model
            mean = self.config["data"]["normalisation"][normalisation_scheme]["mean"]
            std = self.config["data"]["normalisation"][normalisation_scheme]["std"]

            transform = transforms.Compose(
                [
                    transforms.Resize(self.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )

            self.models.append(
                {"model": model, "transform": transform, "name": model_name}
            )

    def predict(self, image):
        """
        Predicts the class of an image using majority voting from the ensemble models.

        Args:
            image (PIL.Image): The input image.

        Returns:
            int: The predicted class index.
        """
        predictions = []

        for item in self.models:
            model = item["model"]
            transform = item["transform"]

            # apply specific transform
            img_tensor = transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = model(img_tensor)
                _, pred = torch.max(output, 1)
                predictions.append(pred.item())

        # majority voting
        vote_counts = Counter(predictions)
        most_common = vote_counts.most_common(1)

        # TODO try out soft voting (probabilities) for ties
        return most_common[0][0]


def train_ensemble_models():
    """
    Trains all models defined in the ensemble configuration.
    """
    config = load_config()
    ensemble_models = config["ensemble2"][
        "models"
    ]  # TODO select ensemble1 or ensemble2

    for model_config in ensemble_models:
        print(f"\n--- Training {model_config['name']} ---")
        train(
            model_name=model_config["name"],
            normalisation_scheme=model_config["normalisation"],
        )


def evaluate_ensemble():
    """
    Evaluates the ensemble classifier on the test dataset and generates a classification report and confusion matrix.
    """
    config = load_config()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(project_root, config["paths"]["img_dir"])
    SPLIT = config["data"]["split"]
    DEVICE = torch.device("mps" if torch.mps.is_available() else "cpu")

    ensemble_config = config["ensemble2"]["models"]  # TODO

    print("\n--- Initialising Ensemble Classifier ---")
    ensemble = EnsembleClassifier(ensemble_config, DEVICE)

    # load the dataset without transform
    print("\n--- Loading Dataset and Splitting ---")
    dataset = BachDataset(root_dir=DATA_DIR, transform=None)

    # get test indices using the same split logic
    _, _, test_dataset = get_stratified_split(dataset, SPLIT)

    print("\n--- Starting Evaluation Loop ---")
    print(f"Evaluating Ensemble on {len(test_dataset)} images…")

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    # simple iteration on test dataset
    for i in tqdm(range(len(test_dataset))):
        image, label = test_dataset[i]  # image is PIL, label is int

        pred = ensemble.predict(image)

        if pred == label:
            correct += 1
        total += 1

        all_preds.append(pred)
        all_labels.append(label)

    accuracy = 100 * correct / total
    print(f"Ensemble Accuracy: {accuracy:.2f}%")

    class_names = config["data"]["classes"]
    print("\n--- Generating Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # confusion matrix
    print("\n--- Creating confusion matrix ---")
    plot_confusion_matrix(
        all_labels,
        all_preds,
        classes=class_names,
        save_path=os.path.join(
            project_root, "results", "ensemble_confusion_matrix.png"
        ),
    )


def evaluate_grand_ensemble(model_configs, k_folds):
    """
    Aggregates predictions from ALL folds of ALL model configurations on the Hold-out Test Set.

    Args:
        model_configs (list): List of model configurations.
        k_folds (int): Number of folds used in training.
    """
    config = load_config()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(project_root, config["paths"]["img_dir"])
    DEVICE = torch.device("mps" if torch.mps.is_available() else "cpu")
    BATCH_SIZE = config["hyperparameters"]["batch_size"]
    LEARNING_RATE = config["hyperparameters"]["learning_rate"]
    IMG_SIZE = tuple(config["hyperparameters"]["img_size"])
    NUM_CLASSES = len(config["data"]["classes"])

    print("\n--- Preparing Grand Ensemble Evaluation ---")

    # prepare test data with same split as in training
    val_transform = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
        ]
    )

    dataset_full = BachDataset(root_dir=DATA_DIR, transform=None)
    _, test_indices = get_holdout_split(dataset_full, test_size=0.1)
    test_subset = Subset(dataset_full, test_indices)

    test_loader = DataLoader(
        test_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=config["execution"]["num_workers"],
    )

    print(f"Evaluating on {len(test_subset)} Hold-out Test samples.")

    # store sum of probabilities for every sample in the test set
    aggregated_probs = torch.zeros(len(test_subset), NUM_CLASSES).to(DEVICE)
    all_targets = []

    # collect targets once
    print("Collecting targets...")
    for images, labels in test_loader:
        all_targets.extend(labels.numpy())
    all_targets = torch.tensor(all_targets).to(DEVICE)

    model_count = 0

    for model_config in model_configs:
        model_name = model_config["name"]
        norm_scheme = model_config["normalisation"]

        mean = config["data"]["normalisation"][norm_scheme]["mean"]
        std = config["data"]["normalisation"][norm_scheme]["std"]

        model_transform = transforms.Compose(
            [
                transforms.Resize(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        for fold in range(k_folds):
            # construct path
            run_folder = f"{model_name}_bs{BATCH_SIZE}_lr{LEARNING_RATE}"
            fold_path = os.path.join(
                project_root, "results", run_folder, f"fold_{fold}", "best_model.pth"
            )

            print(f"Loading {model_name} (Fold {fold}) with {norm_scheme} norm...")

            # initialise model
            if model_name == "resnet18":
                model = ResNet18Model(num_classes=NUM_CLASSES)
            elif model_name == "resnet101":
                model = ResNet101Model(num_classes=NUM_CLASSES)
            elif "densenet121" in model_name:
                model = DenseNet121Model(num_classes=NUM_CLASSES)
            elif "densenet161" in model_name:
                model = DenseNet161Model(num_classes=NUM_CLASSES)

            model.load_state_dict(torch.load(fold_path, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()

            fold_probs = []

            with torch.no_grad():
                for images, _ in test_loader:
                    batch_tensors = []
                    for img in images:
                        pil_img = Image.fromarray(img.numpy())
                        tensor_img = model_transform(pil_img)
                        batch_tensors.append(tensor_img)

                    batch_tensors = torch.stack(batch_tensors).to(DEVICE)

                    outputs = model(batch_tensors)
                    probs = F.softmax(outputs, dim=1)
                    fold_probs.append(probs)

            full_model_probs = torch.cat(fold_probs)
            aggregated_probs += full_model_probs
            model_count += 1

    # calculate final predictions
    avg_probs = aggregated_probs / model_count
    _, final_preds = torch.max(avg_probs, 1)

    # calculate acc
    accuracy = (final_preds == all_targets).float().mean().item() * 100
    print(f"\nGrand Ensemble Accuracy ({model_count} models): {accuracy:.2f}%")

    # classification report
    print("\n--- Classification Report ---")
    print(
        classification_report(
            all_targets.cpu(), final_preds.cpu(), target_names=config["data"]["classes"]
        )
    )

    # confusion matrix
    print("\n--- Confusion Matrix ---")
    plot_confusion_matrix(
        all_targets.cpu(),
        final_preds.cpu(),
        classes=config["data"]["classes"],
        save_path=os.path.join(
            project_root, "results", "grand_ensemble_confusion_matrix.png"
        ),
    )
