import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from src.utils import load_config


class BachDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images (e.g. 'ICIAR2018_BACH_Challenge/Photos').
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        config = load_config()
        self.classes = config["data"]["classes"]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = []
        self.labels = []

        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for img_name in sorted(os.listdir(cls_dir)):
                if img_name.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".tif", ".tiff")
                ):
                    self.images.append(os.path.join(cls_dir, img_name))
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        label = self.labels[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label


def get_stratified_split(dataset, split_ratios, seed=42):
    labels = np.array(dataset.labels)
    num_classes = len(dataset.classes)
    train_indices, val_indices, test_indices = [], [], []

    np.random.seed(seed)

    for i in range(num_classes):
        class_indices = np.where(labels == i)[0]
        np.random.shuffle(class_indices)

        n_total = len(class_indices)
        n_train = int(n_total * split_ratios[0])
        n_val = int(n_total * split_ratios[1])

        train_indices.extend(class_indices[:n_train])
        val_indices.extend(class_indices[n_train : n_train + n_val])
        test_indices.extend(class_indices[n_train + n_val :])

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset


def get_holdout_split(dataset, test_size=0.1, seed=42):
    targets = dataset.labels
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed
    )
    dev_idx, test_idx = next(splitter.split(np.zeros(len(targets)), targets))
    return dev_idx, test_idx


def get_cv_folds(dataset, dev_indices, k_folds=5, seed=42):
    # extract targets specifically for the development subset to ensure stratification
    dev_targets = [dataset.labels[i] for i in dev_indices]

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

    folds = []
    # need to map these back to the original dataset indices
    for relative_train_idx, relative_val_idx in skf.split(
        np.zeros(len(dev_indices)), dev_targets
    ):
        train_idx = dev_indices[relative_train_idx]
        val_idx = dev_indices[relative_val_idx]
        folds.append((train_idx, val_idx))

    return folds
