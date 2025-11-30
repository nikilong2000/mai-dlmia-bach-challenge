import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNet18Model(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNet18Model, self).__init__()
        # resnet18 pretrained on imagenet
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # adjust the last layer to predict four classes
        number_features_last_layer = (
            self.model.fc.in_features
        )  # get number of input features
        self.model.fc = nn.Linear(
            number_features_last_layer, num_classes
        )  # replace the classifier

    def forward(self, x):
        return self.model(x)
