"""
Takes parameters from user; trains, evaluates and saves CNN Image classification model on data.
"""
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
import torchmetrics
import os
import argparse
from data_setup import get_data, get_dataloaders
from models import TinyVGG, get_model_and_weights, transferlearning_prep
from engine import train
from utils import get_devices, set_seeds, save_model, create_summary_writer


parser = argparse.ArgumentParser()
parser.add_argument("--url", type=str, default="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
parser.add_argument("--num_classes", type=int, default=3)
parser.add_argument("--num_epochs", type=int, default=5)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--model", type=str, default=None)
parser.add_argument("--weights", type=str, default=None)
args = parser.parse_args()

data_dir = "./Data"
dataset_dir = "Dataset1"
fname = "temp123.zip"
get_data(data_dir, dataset_dir, args.url, fname)
if args.model is None:
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    model = TinyVGG(n_classes=args.num_classes)
else:
    model, weights = get_model_and_weights(args.model, args.weights)
    transform = weights.transforms()
    transferlearning_prep(model, args.num_classes)

base_dir = os.path.join(data_dir, dataset_dir)
d = get_dataloaders(base_dir, transform, transform, args.batch_size, num_workers=0)
train_dl, val_dl = d["train_dl"], d["val_dl"]

set_seeds(42)
device = get_devices()
model.to(device)
loss_fn = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=args.num_classes).to(device)
writer = create_summary_writer("Experiment", args.model, str(args.num_epochs))
results = train(model, train_dl, val_dl, loss_fn, opt, metric, device, writer, args.num_epochs)
writer.close()
model_name = args.model + str(args.num_epochs)
save_model("./Models", model_name, model)
