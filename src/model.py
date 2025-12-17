import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNet18Model(nn.Module):
    """
    ResNet18 model adapted for the specific number of classes.
    """
    def __init__(self, num_classes=4):
        """
        Initialises the ResNet18 model with pretrained weights and a modified fully connected layer.

        Args:
            num_classes (int, optional): Number of output classes. Defaults to 4.
        """
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
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output logits.
        """
        return self.model(x)


class DenseNet121Model(nn.Module):
    """
    DenseNet121 model adapted for the specific number of classes.
    """
    def __init__(self, num_classes=4):
        """
        Initialises the DenseNet121 model with pretrained weights and a modified classifier layer.

        Args:
            num_classes (int, optional): Number of output classes. Defaults to 4.
        """
        super(DenseNet121Model, self).__init__()
        # Load pretrained DenseNet161
        self.model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

        # densenet uses classifier instead of fc
        number_features_last_layer = self.model.classifier.in_features
        self.model.classifier = nn.Linear(number_features_last_layer, num_classes)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output logits.
        """
        return self.model(x)


class ResNet101Model(nn.Module):
    """
    ResNet101 model adapted for the specific number of classes.
    """
    def __init__(self, num_classes=4):
        """
        Initialises the ResNet101 model with pretrained weights and a modified fully connected layer.

        Args:
            num_classes (int, optional): Number of output classes. Defaults to 4.
        """
        super(ResNet101Model, self).__init__()
        self.model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)

        number_features_last_layer = (
            self.model.fc.in_features
        )  # get number of input features
        self.model.fc = nn.Linear(
            number_features_last_layer, num_classes
        )  # replace the classifier

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output logits.
        """
        return self.model(x)


class DenseNet161Model(nn.Module):
    """
    DenseNet161 model adapted for the specific number of classes.
    """
    def __init__(self, num_classes=4):
        """
        Initialises the DenseNet161 model with pretrained weights and a modified classifier layer.

        Args:
            num_classes (int, optional): Number of output classes. Defaults to 4.
        """
        super(DenseNet161Model, self).__init__()
        # Load pretrained DenseNet161
        self.model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)

        # densenet uses classifier instead of fc
        number_features_last_layer = self.model.classifier.in_features
        self.model.classifier = nn.Linear(number_features_last_layer, num_classes)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output logits.
        """
        return self.model(x)
