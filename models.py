"""
Classes for models to be trained.
"""
import torch
import torch.nn as nn


class TinyVGG(nn.Module):
    """
    Defines a simplified VGG CNN model.
    Attributes:
      conv_block1: 2 Convolutinal layers, followed by a Maxpool.
      conv_block2: 2 Convolutional layers, followed by a Maaxpool.
      classifier: Fully connected section, consisting of a flatten layer and a Linear layer.
    """

    def __init__(self, in_channels: int = 3, n_filters: int = 10, n_classes: int = 3):
        """
        Initializes the CNN model, based on input channels, number of filters in
        Convolutional layers and output dim of final linear layer.
        Args:
          in_channels: Number of channels in input images
          n_filters: (Common) Number of filters in all convolutional layers
          n_classes: Number of classes in the dataset
        """
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=n_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(10 * 16 * 16, n_classes))

    def forward(self, xb):
        return self.classifier(self.conv_block2(self.conv_block1(xb)))

