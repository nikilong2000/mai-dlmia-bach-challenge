import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.utils import load_config


def get_transform(augmentation_strength, normalisation_scheme):
    config = load_config()

    # setting the global variables according to config
    MEAN = config["data"]["normalisation"][normalisation_scheme]["mean"]
    STD = config["data"]["normalisation"][normalisation_scheme]["std"]
    IMG_SIZE = config["hyperparameters"]["img_size"]

    if augmentation_strength == 0:
        # validation transformations (no augmentation)
        baseline_transform = A.Compose(
            [
                A.Resize(height=IMG_SIZE[0], width=IMG_SIZE[1]),
                A.Normalize(
                    mean=MEAN,
                    std=STD,
                ),
                A.ToTensorV2(),
            ]
        )

        return baseline_transform

    elif augmentation_strength == 1:
        geometric_transform_simple = A.Compose(
            [
                A.Resize(height=IMG_SIZE[0], width=IMG_SIZE[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Normalize(
                    mean=MEAN,
                    std=STD,
                ),
                A.ToTensorV2(),
            ]
        )

        return geometric_transform_simple

    elif augmentation_strength == 2:
        geometric_transform_complex = A.Compose(
            [
                A.Resize(height=IMG_SIZE[0], width=IMG_SIZE[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(p=0.5),
                A.ElasticTransform(p=0.5),
                A.Normalize(
                    mean=MEAN,
                    std=STD,
                ),
                A.ToTensorV2(),
            ]
        )

        return geometric_transform_complex
