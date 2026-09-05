import torch.nn as nn
from torchvision import models


def resnet50(pretrained: bool, num_classes: int) -> models.resnet.ResNet:
    model = models.resnet50(
        pretrained=pretrained
    )
    # Grayscale image -> 1 channel
    model.conv1 = nn.Conv2d(
        1,
        64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes)
    return model
