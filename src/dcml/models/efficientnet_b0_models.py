import torch
import torch.nn as nn
from torchvision import models


# def efficientnet_b0(pretrained: bool, num_classes: int):
#     Commented out during cleanup: not selected by current YAML architectures.


# def efficientnet_b0_1(pretrained: bool, num_classes: int):
#     Commented out during cleanup: not selected by current YAML architectures.


# def efficientnet_b0_2(pretrained: bool, num_classes: int):
#     Commented out during cleanup: not selected by current YAML architectures.


# def efficientnet_b0_3(pretrained: bool, num_classes: int):
#     Commented out during cleanup: not selected by current YAML architectures.


# def efficientnet_b0_4(pretrained: bool, num_classes: int):
#     Commented out during cleanup: not selected by current YAML architectures.


def efficientnet_b0_5(pretrained: bool, num_classes: int):
    """
                First layer 1x is initialized from pretrained 3x layer by taking average over 3 channels.
                The four last (6, 7, 8, 9th) blocks are cut out
    """
    if pretrained:
        weights = "IMAGENET1K_V1"
    else:
        weights = None

    model = models.efficientnet_b0(weights=weights)

    # weights_rgb = model.features[0][0].weight
    with torch.no_grad():
        avg_rgb_weights = torch.mean(model.features[0][0].weight, dim=1)

    # Grayscale image -> 1 channel
    model.features[0][0] = nn.Conv2d(
        1,
        32,
        kernel_size=3,
        stride=2,
        padding=1,
        bias=False
    )

    with torch.no_grad():
        model.features[0][0].weight[:, 0, :, :] = avg_rgb_weights

    model.features = model.features[:-4]
    model.features.add_module('AddedSiLu', nn.SiLU(inplace=True))

    model.classifier[1] = nn.Linear(80, num_classes)
    return model
