from collections import OrderedDict


def train_one_epoch(device, model, criterion, optimizer, train_loader, epoch, scheduler):
    model.train()
    total_loss = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        pred = model(data)

        # Hack to get output from deeplabv3_resnet50
        if isinstance(pred, OrderedDict):
            pred = pred['out']

        loss = criterion(pred, target)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if scheduler is not None:
        scheduler.step()

    loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}: Loss: {loss}")
    return loss
