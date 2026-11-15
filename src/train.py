"""
Module: src.train
Training, validation, regularization, and optimization routines for Retinal OCT models.
Supports Early Stopping, L2 Regularization, and Learning Rate Scheduling.
"""

import copy
import time
import torch
import torch.nn as nn
from tqdm import tqdm


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Executes a single training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


def validate_one_epoch(model, dataloader, criterion, device):
    """Executes a single validation epoch without updating weights."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=15,
    lr=1e-4,
    weight_decay=1e-4,
    patience=5,
    device=None,
    model_name="Model"
):
    """
    Trains a model with Early Stopping, AdamW optimizer with L2 weight decay,
    and ReduceLROnPlateau scheduler.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []
    }

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    print(f"\n--- Training {model_name} on {device} (Epochs: {num_epochs}, LR: {lr}, L2: {weight_decay}) ---")
    start_time = time.time()

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch+1:02d}/{num_epochs:02d} | "
            f"Train Loss: {train_loss:.4f} - Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} - Acc: {val_acc:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1} (No improvement for {patience} epochs).")
                break

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f}s. Restoring best model weights (Best Val Loss: {best_val_loss:.4f}).")
    model.load_state_dict(best_weights)
    return model, history
