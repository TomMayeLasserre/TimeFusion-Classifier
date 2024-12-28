import torch
from tqdm import tqdm


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total = 0
    for X, y in tqdm(train_loader, desc="Training"):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        preds = out.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total += X.size(0)
    return total_loss/total, total_correct/total


def eval_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y)
            total_loss += loss.item() * X.size(0)
            preds = out.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total += X.size(0)
    return total_loss/total, total_correct/total
