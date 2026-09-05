import torch
import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self, pretrained:bool, num_classes: int):
        super(LeNet5, self).__init__()

        if pretrained:
            raise NotImplementedError("Pretrained weights are not available for LeNet5.")

        self.conv_features = 16
        self.n_features = self.conv_features * 7 * 17

        # Convolutional layers
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels=6, out_channels=self.conv_features, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((7, 17)) # corresponds to 40 x 80 input images
        )

        # Fully connected layers
        self.fc_block = nn.Sequential(
            nn.Linear(in_features=self.n_features, out_features=120),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=84, out_features=num_classes)
        )

    def forward(self, x):
        x = self.conv_block(x)
        # Flatten the output for the fully connected layers
        x = torch.flatten(x, 1)
        x = self.fc_block(x)
        return x
