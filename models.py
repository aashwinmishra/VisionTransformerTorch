"""
Classes for models to be trained.
"""
import torch
import torch.nn as nn
import torchvision
from typing import Tuple


class PatchEmbeddingConv(torch.nn.Module):
  """
  Defines the alternate Patch Embeddings via Conv layers in Dosovitskiy et al (2021).
  Attributes:
      layer: Convolutional layer to create patches and embeddings.
      flatten: Flatten layer to flatten p \time p patches to a p^2 vector.
  """
  def __init__(self, in_channels: int = 3, patch_size: int = 16, embedding_dim: int = 768):
    """
    Args:
      :param in_channels: Channels in images. 3 if color images, 1 if BnW
      :param patch_size: Size of each square patch.
      :param embedding_dim: Dimension of embedding space.
    """
    super().__init__()
    self.layer = torch.nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size)
    self.flatten = torch.nn.Flatten(start_dim=2)

  def forward(self, xb):
    assert xb.shape[-1] % self.layer.stride == 0, "Images need to be patched perfectly."
    return self.flatten(self.layer(xb)).permute(0, 2, 1)


class ClassEmbedding(torch.nn.Module):
  """
  Defines a learnable class embedding for the ViT.
  Attributes:
    class_embedding: Parameter of size [batch_size, 1, embedding_dimension]
  """
  def __init__(self, batch_size: int=64, embedding_dim: int=768):
    super().__init__()
    self.class_embedding = torch.nn.Parameter(torch.rand(batch_size,1,embedding_dim))

  def forward(self, xb):
    return torch.cat([self.class_embedding, xb], dim=1)


class PositionalEncoding(torch.nn.Module):
  """
  Defines a learnable positional encoding for the ViT.
  Attributes:
    positional_embedding: Parameter of size [1, sequence_length, embedding_dimension]
  """
  def __init__(self, sequence_length: int=197, embedding_dim: int=768):
    super().__init__()
    self.positional_embedding = torch.nn.Parameter(torch.randn(1,sequence_length, embedding_dim))

  def forward(self, xb):
    return xb + self.positional_embedding


class PatchEmbedding(nn.Module):
  def __init__(self, p: int=16, N: int=3136, D: int = 128):
    super().__init__()
    self.p = p
    self.N = N
    self.unfold = nn.Unfold(kernel_size=(p,p), stride=p) #Takes Images of [N, 3, 224, 224] -> [N, L, 3*p^2]. Needs permutation of axes.
    self.embedding = nn.Linear(p*p*3, D) #Cheap learnable Embedding.

  def forward(self, x):
    x = self.unfold(x)
    x = x.permute(0, 2, 1)
    return self.embedding(x)


def get_model_and_weights(model_name: str, weights_version: str = None) -> Tuple:
  """
  Takes specifications of the pretrained model and the weights to be loaded.
  Returns a tuple with model and weights.
  Args:
    model_name: Name and version of pretrained model to be loaded from torchvision.models.
    weights_version: version of weights for the pretrianed model.
  Returns:
    Tuple with pretrained model and weights.
  Raises:
    KeyError: if the model_name is not found in the list of image classification models at torchvision.models.
  """
  all_models = torchvision.models.list_models(module=torchvision.models)
  if model_name.lower() not in all_models:
    raise KeyError
  else:
    if weights_version is None:
      weights_version="DEFAULT"
    model = torchvision.models.get_model(model_name, weights=weights_version)
    weights_config = model_name+"_Weights."+weights_version
    weights = torchvision.models.get_weight(weights_config)
    return model, weights


def transferlearning_prep(model: torch.nn.Module, num_classes: int):
  """
  Prepares pre-trained model for training, by
  freezing feature extractor weights and correcting final layer for number
  of classes in our dataset.
  Args:
    model: pretrained model
    num_classes: number of classes in current dataset
  Returns:
    None
  Raises:
    NameError: if the pre-trained model does not have a features or a
    classifier section.
  """
  try:
    for param in model.features.parameters():
      param.requires_grad = False
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 3)
  except:
    raise NameError


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

