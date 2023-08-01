import torch
from collections import OrderedDict


def validate(device, model, criterion, loader, iou=None):
    model.eval()
    loss = 0.
    iou_list = []
    n_samples = 0
    with torch.no_grad():
        for (data, target) in loader:
            data, target = data.to(device), target.to(device)
            pred = model(data)

            # Hack to get output from deeplabv3_resnet50
            if isinstance(pred, OrderedDict):
                pred = pred['out']

            loss += criterion(pred, target).item()
            n_samples += len(data)

            if iou is not None:
                pred_masks = torch.argmax(pred, dim=1).unsqueeze(1).to('cpu')
                target_masks = torch.argmax(target, dim=1).unsqueeze(1).to('cpu')
                iou_list.append(iou(pred_masks, target_masks))

    return loss / len(loader), sum(iou_list) / n_samples
