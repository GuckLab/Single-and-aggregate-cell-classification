import torch
import torch.nn as nn
from torchvision import models


def efficientnet_b0_5(pretrained):
    """
                First layer 1x is initialized from pretrained 3x layer by taking average over 3 channels.
                The four last (6, 7, 8, 9th) blocks are cut out
    """
    if pretrained:
        weights = "IMAGENET1K_V1"
    else:
        weights = None

    model = models.efficientnet_b0(weights=weights)

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

    model.classifier = nn.Identity()
    return model


class MT_EfficientNet(nn.Module):
    def __init__(self, pretrained, num_classes, num_heads):
        super(MT_EfficientNet, self).__init__()

        self.num_heads = num_heads
        self.num_mc_classes = num_classes-self.num_heads
        self.backbone = efficientnet_b0_5(pretrained=pretrained)
        self.n_features = 80  # TODO get it automatically None

        self.mtl_fc_hidden1 = 20

        self.FCN = nn.Sequential(
            nn.Linear(self.n_features * 1 * 1, self.num_heads),
            nn.ReLU(inplace=True),
        )

        self.heads = nn.ModuleList([])
        for _ in range(self.num_heads):
            self.heads.append(
                nn.Sequential(
                nn.Linear(self.num_heads, self.num_heads),
                nn.ReLU(True),
                nn.Linear(self.num_heads, 1),
                )
            )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.n_features, self.num_mc_classes)
            )

    def forward(self, x):

        # MTL
        features = self.backbone(x)

        features_MTL = self.FCN(features)
        outputs = torch.empty(features_MTL.shape[0], self.num_heads, device=features_MTL.device)
        for n, head in enumerate(self.heads):
            outputs[:, n] = head(features_MTL).squeeze()

        # multi-class
        cls = self.classifier(features)

        out = torch.cat((outputs, cls), dim=1)

        return out




