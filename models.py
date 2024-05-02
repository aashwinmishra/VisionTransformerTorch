"""
Classes for models to be trained.
"""
import torch
import torch.nn as nn
import torchvision
from typing import Tuple


class PatchEmbedding(torch.nn.Module):
    """
     Defines the alternate Patch Embeddings via Conv layers in Dosovitskiy et al (2021).
     Attributes:
         layer: Convolutional layer to create patches and embeddings.
         flatten: Flatten layer to flatten p \time p patches to a p^2 vector.
    """
    def __init__(self, in_channels: int = 3, patch_size: int = 16, embedding_dim: int = 768):
        super().__init__()
        self.layer = torch.nn.Conv2d(in_channels=in_channels,
                                     out_channels=embedding_dim,
                                     kernel_size=patch_size,
                                     stride=patch_size)
        self.flatten = torch.nn.Flatten(start_dim=2)

    def forward(self, xb):
        return self.flatten(self.layer(xb)).permute(0, 2, 1)


class ClassEmbedding(torch.nn.Module):
    """
    Defines a learnable class embedding for the ViT.
    Attributes:
      class_embedding: Parameter of size [batch_size, 1, embedding_dimension]
    """
    def __init__(self, batch_size: int = 64, embedding_dim: int = 768):
        super().__init__()
        self.class_embedding = torch.nn.Parameter(torch.rand(batch_size, 1, embedding_dim))

    def forward(self, xb):
        return torch.cat([self.class_embedding, xb], dim=1)


class PositionalEncoding(torch.nn.Module):
    """
    Defines a learnable positional encoding for the ViT.
    Attributes:
      positional_embedding: Parameter of size [1, sequence_length, embedding_dimension]
    """

    def __init__(self, sequence_length: int = 197, embedding_dim: int = 768):
        super().__init__()
        self.positional_embedding = torch.nn.Parameter(torch.randn(1, sequence_length, embedding_dim))

    def forward(self, xb):
        return xb + self.positional_embedding


class MSA(torch.nn.Module):
    """
    Defines the Multi-head Self Attention operation from Dosovitskiy et al (2021), with the ViT-Base Hyperparameters.
    Attributes:
      msa: Multihead Self Attention Layer
    """
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, attn_dropout: float = 0.0):
        super().__init__()
        self.msa = torch.nn.MultiheadAttention(embed_dim=embed_dim,
                                               num_heads=num_heads,
                                               dropout=attn_dropout,
                                               batch_first=True)

    def forward(self, xb):
        attn_output, attn_output_weights = self.msa(xb, xb, xb)
        return attn_output


class MLP(torch.nn.Module):
    """
    Defines the MLP operation from Dosovitskiy et al (2021), with the ViT-Base Hyperparameters.
    Attributes:
     mlp: MLP layers, with GeLU and Dropout
    """
    def __init__(self, embed_dim: int = 768,
                 mlp_size: int = 3072,
                 mlp_dropout: float = 0.1):
        super().__init__()
        self.mlp = torch.nn.Sequential(nn.Linear(embed_dim, mlp_size),
                                       nn.GELU(),
                                       nn.Dropout(mlp_dropout),
                                       nn.Linear(mlp_size, embed_dim),
                                       nn.Dropout(mlp_dropout))

    def forward(self, xb):
        return self.mlp(xb)


class PreNorm(nn.Module):
    """
    Defines the Pre-Layer Norm operation from Dosovitskiy et al (2021).
    Attributes:
        norm: Layer Norm
        fn: Function to be applied to input after the Layer norm's application
    """
    def __init__(self, embed_dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fn = fn

    def forward(self, xb):
        return self.fn(self.norm(xb))


class ResidualAdd(nn.Module):
    """
    Defines the skip-connection from Dosovitskiy et al (2021).
    Attributes:
        fn: function to be applied, to which the skip connection is added.
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, xb):
        return self.fn(xb) + xb


class ViT(nn.Module):
    """
    Defines the ViT-Base model from Dosovitskiy et al (2021).
    Attributes:
        patch_embedding: Converts image into a sequence of patches.
        class_embedding: pre-pends the learnable class token
        positional_encoding: Adds the learnable positional encoding
        layers: ModuleList of Multiple Transformer Encoder Blocks.
        head: MLP head from Dosovitskiy et al (2021) (ie, equation 4)
    """
    def __init__(self,
                 in_channels: int = 3,
                 image_size: int = 224,
                 patch_size: int = 16,
                 num_encoders: int = 12,
                 embed_dim: int = 768,
                 mlp_size: int = 3072,
                 num_heads: int = 12,
                 attn_dropout: float = 0.0,
                 mlp_dropout: float = 0.1,
                 num_classes: int = 3):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embed_dim)
        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embed_dim))
        self.positional_encoding = PositionalEncoding(num_patches + 1, embed_dim)

        self.layers = nn.ModuleList([])
        for _ in range(num_encoders):
            transformer_block = nn.Sequential(
                ResidualAdd(PreNorm(embed_dim, MSA(embed_dim, num_heads, attn_dropout))),
                ResidualAdd(PreNorm(embed_dim, MLP(embed_dim, mlp_size, mlp_dropout)))
            )
            self.layers.append(transformer_block)

        self.head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embedding(x)
        class_token = self.class_embedding.expand(batch_size, -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = self.positional_encoding(x)

        for i in range(len(self.layers)):
            x = self.layers[i](x)

        return self.head(x[:, 0, :])


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


def transferlearning_prep_ViT(model: torch.nn.Module, num_classes: int):
  """
  Prepares pre-trained ViT for training, by
  freezing all weights and replacing the final layer for number
  of classes in our dataset.
  Args:
    model: pretrained ViT
    num_classes: number of classes in current dataset
  Returns:
    None
  Raises:
    NameError: if the pre-trained model does not have a heads section.
  """
  try:
    for param in model.parameters():
      param.requires_grad = False
    model.heads[-1] = nn.Linear(model.heads[-1].in_features, 3)
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

