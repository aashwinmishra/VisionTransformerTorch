"""
Functions to train and evaluate model on the image dataset
"""
import torch
import torch.nn as nn
import torchmetrics
from typing import Dict, List, Tuple


def train_step(model: torch.nn.Module,
               train_dl: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               opt: torch.optim.Optimizer,
               metric: torchmetrics.metric,
               device: torch.device) -> Dict:
    """
    Performs 1 epoch of training of model on train dataloader,
    returning model loss and accuracy over the epooch.
    Args:
      model: model too be trained
      train_dl: Dataloader with training data
      loss_fn: Differentiable loss function to be used for gradients
      opt: Optimizer to train model.
      metric: Metric to evaluate model performance.
      device: Device on which model and data will reside.
    Returns:
      Dict with keys "epoch_loss" and "epoch_metric"
    """

    model.train()
    losses = []
    for images, labels in train_dl:
        images, labels = images.to(device), labels.to(device)
        opt.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        opt.step()
        batch_metric = metric.update(outputs.softmax(dim=-1), labels)
        losses.append(loss.detach().cpu().item())
    epoch_metric = metric.compute()
    metric.reset()
    epoch_loss = sum(losses) / len(losses)
    model.eval()
    return {"epoch_loss": epoch_loss, "epoch_metric": epoch_metric}


def val_step(model: torch.nn.Module,
             val_dl: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             metric: torchmetrics.metric,
             device: torch.device) -> Dict:
    """
    Performs one epoch of evaluation of the model on the validation data,
    returning validation loss and metric over the epoch.
    Args:
      model: model to be evaluated.
      val_dl: Dataloader with validation dataset
      loss_fn: Differentiable loss function.
      metric: Metric to evaluate model performance.
      device: Device on which model and data will reside.
    Returns:
      Dict with keys "epoch_loss" and "epoch_metric"
    """
    model.eval()
    losses = []
    with torch.inference_mode():
        for images, labels in val_dl:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            batch_metric = metric.update(outputs.softmax(dim=-1), labels)
            losses.append(loss.detach().cpu().item())
    epoch_metric = metric.compute()
    metric.reset()
    epoch_loss = sum(losses) / len(losses)
    return {"epoch_loss": epoch_loss, "epoch_metric": epoch_metric}


def train(model: torch.nn.Module,
          train_dl: torch.utils.data.DataLoader,
          val_dl: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          opt: torch.optim.Optimizer,
          metric: torchmetrics.metric,
          device: torch.device,
          num_epochs: int = 10) -> Dict:
    """
    Performs defined number of epochs of trianing and validation for the model on
    the data loaders, returning the losses and metrics for each epoch.
    Args:
      model: model to be trained and evaluated.
      train_dl: Dataloader with training data.
      val_dl: Dataloader with testing data.
      loss_fn: Differentiable loss function to use for gradients.
      opt: Optimizer to tune model params.
      metric: Metric to evaluate model.
      device: Device on which model and eventually data shall reside
      num_epochs: Number of epochs of training
    Returns:
      Dict with history of losses and metric on train and val data.
    """
    train_losses, val_losses, train_metrics, val_metrics = [], [], [], []
    for epoch in range(num_epochs):
        train_result = train_step(model, train_dl, loss_fn, opt, metric, device)
        val_result = val_step(model, val_dl, loss_fn, metric, device)
        train_losses.append(train_result["epoch_loss"])
        train_metrics.append(train_result["epoch_metric"])
        val_losses.append(val_result["epoch_loss"])
        val_metrics.append(val_result["epoch_metric"])
        print(
            f"Epoch: {epoch + 1} Train Loss: {train_losses[-1]} Train Metric: {train_metrics[-1]} Val Loss: {val_losses[-1]} Val Metirc: {val_metrics[-1]}")

    return {"train_losses": train_losses, "train_metrics": train_metrics,
            "val_losses": val_losses, "val_mettrics": val_metrics}



