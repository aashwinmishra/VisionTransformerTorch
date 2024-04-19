"""
Classes to define dataset and dataloader.
Functions to download data, create datasets and dataloaders.
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.io import read_image
import os
import requests
import zipfile
import pathlib
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple


class ImageClassificationDataset(torch.utils.data.Dataset):
    """
    Defines a simple torch dataset for image classification, subclassing the
    Dataset class.
    Attributes:
      paths: paths to all images in the dataset
      transform: transform to be applied to images
      class_names: names of the classes
      class_to_idx: dictionary mapping class names to corresponding indices.
    """

    def __init__(self, img_dir: str, transform: torchvision.transforms = None):
        self.paths = list(pathlib.Path(img_dir).glob("*/*.jpg"))
        self.transform = transform
        self.class_names, self.class_to_idx = self.find_classes(img_dir)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        img_path = self.paths[idx]
        image = Image.open(img_path)
        class_name = self.paths[idx].parent.name
        label = self.class_to_idx[class_name]
        if self.transform:
            image = self.transform(image)
        return image, label

    @staticmethod
    def find_classes(img_dir):
        class_names = sorted(os.listdir(img_dir))
        class_to_idx = {}
        for idx, name in enumerate(class_names):
            class_to_idx[name] = idx
        return class_names, class_to_idx


def get_data(data_dir: str,
             dataset_dir: str,
             url: str,
             fname: str) -> None:
    """
    Downloads data from url, unzips it and stores it in ImageFolder format to data_dir/dataset_dir.
    Args:
      data_dir: Directory with all datasets
      dataset_dir: sub-directory for this dataset
      url: web address to download from
      fname: name fo temporary file.
    Returns:
      None
    """
    dataset_dir = os.path.join(data_dir, dataset_dir)
    if Path(dataset_dir).is_dir():
        print("Dataset Exists. Skipping Download.")
    else:
        os.makedirs(dataset_dir, exist_ok=True)
        temp_file = os.path.join(data_dir, fname)

        with open(temp_file, "wb") as f:
            request = requests.get(url)
            f.write(request.content)

        with zipfile.ZipFile(os.path.join(data_dir, fname), "r") as zip_file:
            zip_file.extractall(dataset_dir)

        os.remove(temp_file)


def get_dataloaders(base_dir: str,
                    train_transforms: torchvision.transforms,
                    val_transforms: torchvision.transforms,
                    batch_size: int,
                    num_workers: int) -> Dict:
    """
    Takes base_dir with data in ImageFolder format. Creates Dataset and Datalooaders.
    Returns train and val dataloaders in a dict.
    Args:
      base_dir: dataset directory, with data stored in ImageFolder format.
      batch_size: batch size of the dataloader
      num_workers: num of cores to assign to loader.
    Returns:
      Dictionary with keys "train_dl" and "val_dl".
    """
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "test")

    train_ds = ImageClassificationDataset(train_dir, transform=train_transforms)
    val_ds = ImageClassificationDataset(val_dir, transform=val_transforms)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)

    return {"train_dl": train_dl, "val_dl": val_dl}
