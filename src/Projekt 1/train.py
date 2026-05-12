import torch



def run_epoch(model, loader, criterion, optimizer=None):

    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0

    total_correct = 0

    total_samples = 0

    for xb, yb in loader:

        if is_train:
            optimizer.zero_grad()

        logits = model(xb)

        loss = criterion(logits, yb)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)

        total_correct += (preds == yb).sum().item()

        total_samples += yb.size(0)

    accuracy = total_correct / total_samples

    return total_loss / len(loader), accuracy