import os
import numpy as np
from torch.utils.data import Dataset, Subset
from PIL import Image

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
            for img_name in os.listdir(cls_dir):
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
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_stratified_split(dataset, split_ratios, seed=42):
    labels = np.array(dataset.labels)
    num_classes = len(dataset.classes)
    train_indices, val_indices, test_indices = [], [], []

    # fixed seed for reproducibility of the shuffle
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
