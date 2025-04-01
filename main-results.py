import random
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from model import MPNNSyn
from utils import MyDataset, collate
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# Set random seeds for reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def train(model, device, loader_train, optimizer, loss_fn):
    model.train()
    epoch_losses = []
    for batch_idx, (data1, data2, y) in enumerate(loader_train):
        data1, data2, y = data1.to(device), data2.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(data1, data2)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    return np.mean(epoch_losses)

def predicting(model, device, loader_test):
    model.eval()
    total_preds = []
    total_labels = []
    with torch.no_grad():
        for data1, data2, y in loader_test:
            data1, data2 = data1.to(device), data2.to(device)
            output = model(data1, data2)
            ys = F.softmax(output, dim=1).cpu().numpy()
            total_preds.extend(ys[:, 1])
            total_labels.extend(y.numpy())
    return np.array(total_labels), np.array(total_preds)

def plot_combined(epochs, accuracies, losses, fold):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, losses, color=color, alpha=0.5)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(epochs, accuracies, color=color, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Loss vs. Accuracy (Fold {fold})')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f'figures/fold{fold}_combined.png')
    plt.close()

def plot_accuracy_drops(epochs, accuracies, fold):
    drops = []
    for i in range(1, len(accuracies)):
        if accuracies[i] < accuracies[i-1] - 0.03:
            drops.append(epochs[i])

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, accuracies, 'b-', linewidth=2)
    
    for drop in drops:
        plt.axvline(x=drop, color='r', linestyle='--', alpha=0.5)
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy Drops >3% (Fold {fold})')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(0.5, 0.9)
    plt.savefig(f'figures/fold{fold}_drops.png')
    plt.close()

# Main configuration
TRAIN_BATCH_SIZE = 128
LR = 0.0005
NUM_EPOCHS = 200
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create output directory
os.makedirs('figures', exist_ok=True)

# Dataset setup
dataset = MyDataset()
lenth = len(dataset)
pot = int(lenth / 5)
random_num = list(range(len(dataset)))
random.shuffle(random_num)

# Metrics storage
all_epoch_accuracies = []
all_epoch_losses = []

for fold in range(5):
    test_indices = random_num[pot*fold : pot*(fold+1)]
    train_indices = random_num[:pot*fold] + random_num[pot*(fold+1):]
    
    loader_train = DataLoader(
        dataset.get_data(train_indices),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate
    )
    loader_test = DataLoader(
        dataset.get_data(test_indices),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate
    )

    model = MPNNSyn().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    fold_accuracies = []
    fold_losses = []
    
    for epoch in range(NUM_EPOCHS):
        epoch_loss = train(model, device, loader_train, optimizer, loss_fn)
        true_labels, probs = predicting(model, device, loader_test)
        acc = accuracy_score(true_labels, (probs > 0.5).astype(int))
        
        fold_accuracies.append(acc)
        fold_losses.append(epoch_loss)
        
        print(f"Fold {fold+1} | Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Acc: {acc:.4f}")

    all_epoch_accuracies.append(fold_accuracies)
    all_epoch_losses.append(fold_losses)

    # Plot fold-specific figures
    epochs = list(range(1, NUM_EPOCHS+1))
    
    # Accuracy Progression
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, fold_accuracies, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy Progression (Fold {fold+1})')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(0.5, 0.9)
    plt.savefig(f'figures/fold{fold+1}_accuracy.png')
    plt.close()

    # Training Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, fold_losses, 'r-', linewidth=1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss (Fold {fold+1})')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f'figures/fold{fold+1}_loss.png')
    plt.close()

    # Combined Plot
    plot_combined(epochs, fold_accuracies, fold_losses, fold+1)

    # Accuracy Drops
    plot_accuracy_drops(epochs, fold_accuracies, fold+1)

# Generate cross-fold average figures
mean_accuracies = np.mean(all_epoch_accuracies, axis=0)
epochs = list(range(1, NUM_EPOCHS+1))

# Mean Accuracy Progression
plt.figure(figsize=(10, 5))
plt.plot(epochs, mean_accuracies, 'b-', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Mean Accuracy Progression (All Folds)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.ylim(0.5, 0.9)
plt.savefig('figures/mean_accuracy.png')
plt.close()

print("\nAll figures saved to 'figures' directory:")
print("- [foldX]_accuracy.png: Individual fold accuracy progression")
print("- [foldX]_loss.png: Individual fold training loss")
print("- [foldX]_combined.png: Loss-accuracy comparison")
print("- [foldX]_drops.png: Accuracy drops visualization")
print("- mean_accuracy.png: Cross-fold average accuracy progression")