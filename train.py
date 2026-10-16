"""Train the exact Assignment 2 CNN on CIFAR-10.

Local: python train.py --epochs 10
Colab: !python train.py --epochs 10 --batch-size 128
"""
import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from app.cnn_model import CNNClassifier


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--output", default="models/cnn_cifar10.pth")
    args = p.parse_args()
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}", flush=True)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    training = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    testing = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
    train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(testing, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = CNNClassifier().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    for epoch in range(args.epochs):
        model.train()
        total_loss, correct, seen = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            scores = model(inputs)
            loss = loss_fn(scores, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (scores.argmax(1) == labels).sum().item()
            seen += labels.size(0)
        print(f"Epoch {epoch + 1}/{args.epochs}: loss={total_loss / seen:.4f}; "
              f"train accuracy={correct / seen:.2%}", flush=True)
    model.eval()
    right, count = 0, 0
    with torch.inference_mode():
        for inputs, labels in test_loader:
            scores = model(inputs.to(device))
            right += (scores.argmax(1).cpu() == labels).sum().item()
            count += labels.size(0)
    print(f"CIFAR-10 test accuracy: {right / count:.2%}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), output)
    print(f"Model saved: {output.resolve()}")


if __name__ == "__main__":
    main()
