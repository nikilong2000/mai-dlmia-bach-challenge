import torch
import os
import numpy as np
from collections import Counter
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report

from src.model import ResNet101Model, DenseNet161Model, ResNet18Model
from src.utils import load_config
from src.dataset import BachDataset, get_stratified_split
from src.train import train
from src.visualisations import plot_confusion_matrix


class EnsembleClassifier:
    def __init__(self, models_config, device):
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

        # if its a tie, most_common(1) just picks the first one encountered
        # TODO try out soft voting (probabilities) for ties
        return most_common[0][0]


def train_ensemble_models():
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

    # simple iteration is fine for evaluation on test dataset
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
