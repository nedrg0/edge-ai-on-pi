"""
MNIST On-Device Training Demo
==============================
A simple neural network trained directly on the Raspberry Pi.
Demonstrates edge AI training inspired by the POET paper.

Requirements:
    pip install torch torchvision

Run:
    python train_mnist_rpi.py
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
EPOCHS      = 3          # Keep low on Pi to save time (3–5 recommended)
BATCH_SIZE  = 32         # Smaller batch = less memory (important on edge devices)
LEARNING_RATE = 0.001
DATA_DIR    = "./data"   # MNIST will be downloaded here automatically


# ─────────────────────────────────────────────
# 2. DEVICE SETUP
# ─────────────────────────────────────────────
# Raspberry Pi does not have a GPU, so we always use CPU
device = torch.device("cpu")
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}\n")


# ─────────────────────────────────────────────
# 3. DATA LOADING
# ─────────────────────────────────────────────
# Transform: convert images to tensors and normalize pixel values to [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download MNIST dataset (only needed once; cached to DATA_DIR after that)
print("Loading MNIST dataset...")
train_dataset = datasets.MNIST(root=DATA_DIR, train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)

# DataLoader wraps the dataset and handles batching
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Training samples : {len(train_dataset)}")
print(f"Test samples     : {len(test_dataset)}\n")


# ─────────────────────────────────────────────
# 4. MODEL DEFINITION
# ─────────────────────────────────────────────
# A small fully-connected network — intentionally tiny to run well on the Pi
class SmallNet(nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()
        # Input: 28x28 = 784 pixels (flattened)
        self.fc1 = nn.Linear(784, 128)   # Hidden layer 1
        self.fc2 = nn.Linear(128, 64)    # Hidden layer 2
        self.fc3 = nn.Linear(64, 10)     # Output: 10 classes (digits 0–9)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)   # Flatten the 28x28 image into a 784-dim vector
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)        # No softmax here — CrossEntropyLoss handles it
        return x

model = SmallNet().to(device)
print("Model architecture:")
print(model)

# Count total parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}\n")


# ─────────────────────────────────────────────
# 5. LOSS FUNCTION & OPTIMIZER
# ─────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()          # Standard loss for classification
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ─────────────────────────────────────────────
# 6. TRAINING LOOP
# ─────────────────────────────────────────────
def train_one_epoch(epoch):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0
    start_time = time.time()

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          # Clear gradients from previous step
        outputs = model(images)        # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()                # Backward pass (compute gradients)
        optimizer.step()               # Update weights

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

        # Print progress every 200 batches
        if (batch_idx + 1) % 200 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(train_loader)} "
                  f"| Loss: {total_loss/(batch_idx+1):.4f} "
                  f"| Acc: {100.*correct/total:.2f}%")

    elapsed = time.time() - start_time
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    print(f"\n→ Epoch {epoch} complete | Loss: {avg_loss:.4f} | "
          f"Train Acc: {accuracy:.2f}% | Time: {elapsed:.1f}s\n")
    return avg_loss, accuracy


# ─────────────────────────────────────────────
# 7. EVALUATION
# ─────────────────────────────────────────────
def evaluate():
    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():   # No gradient computation needed during evaluation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)

    accuracy = 100. * correct / total
    print(f"✓ Test Accuracy: {accuracy:.2f}%\n")
    return accuracy


# ─────────────────────────────────────────────
# 8. RUN TRAINING
# ─────────────────────────────────────────────
print("=" * 50)
print(f"Starting training for {EPOCHS} epoch(s)...")
print("=" * 50 + "\n")

history = {"train_loss": [], "train_acc": [], "test_acc": []}

total_start = time.time()
for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(epoch)
    test_acc = evaluate()
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["test_acc"].append(test_acc)

total_time = time.time() - total_start

print("=" * 50)
print("Training complete!")
print(f"Total time   : {total_time:.1f}s")
print(f"Final Test Acc: {history['test_acc'][-1]:.2f}%")
print("=" * 50)


# ─────────────────────────────────────────────
# 9. SAVE THE MODEL
# ─────────────────────────────────────────────
MODEL_PATH = "mnist_model.pth"
torch.save(model.state_dict(), MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")
print("To reload it later:")
print("  model = SmallNet()")
print("  model.load_state_dict(torch.load('mnist_model.pth'))")


# ─────────────────────────────────────────────
# 10. MEMORY USAGE REPORT
# ─────────────────────────────────────────────
# Relevant to the POET paper — shows how much memory training uses
try:
    import psutil, os
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"\nMemory used by this process: {mem_mb:.1f} MB")
    print("(This is what POET aims to minimize on constrained devices)")
except ImportError:
    print("\nInstall psutil to see memory usage: pip install psutil")
