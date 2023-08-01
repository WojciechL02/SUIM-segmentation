import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from torchmetrics.classification import MulticlassJaccardIndex

from dataset import SUIMDataset
from train import train_one_epoch
from test import validate
from model.unet import UNet
from utils import decode_segmap, compare_masks, plot_loss

import argparse
from collections import OrderedDict


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, choices=["unet", "deeplabv3"], default="unet", help="which model to use")
    args = parser.parse_args()

    # === Setup dataset ===
    t = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]
    )
    t_mask = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.PILToTensor()
        ]
    )

    train_data = SUIMDataset("./data/train_val", t, t_mask)
    test_data = SUIMDataset("./data/test", t, t_mask)

    N_CLASSES = 8
    BATCH_SIZE = 16
    LR = 0.001
    WEIGHT_DECAY = 1e-4
    EPOCHS = 50

    indices = list(range(len(train_data)))
    np.random.shuffle(indices)
    split = int(np.floor(0.2 * len(train_data)))
    train_sample = SubsetRandomSampler(indices[split:])
    val_sample = SubsetRandomSampler(indices[:split])

    train_loader = DataLoader(train_data, sampler=train_sample, batch_size=BATCH_SIZE)
    val_loader = DataLoader(train_data, sampler=val_sample, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=BATCH_SIZE)

    # === Setup model ===
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, criterion, optimizer = None, None, None

    if (args.model == "unet"):
        model = UNet(n_classes=N_CLASSES).to(device)

    elif (args.model == "deeplabv3"):
        model = deeplabv3_resnet50(num_classes=N_CLASSES).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    iou = MulticlassJaccardIndex(N_CLASSES)

    # TRAINING LOOP
    best_loss = 3.0
    loss_history = {"train": [], "val": []}
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(device, model, criterion, optimizer, train_loader, epoch, scheduler)
        val_loss, _ = validate(device, model, criterion, val_loader)
        loss_history["train"].append(train_loss)
        loss_history["val"].append(val_loss)
        if val_loss < best_loss:
            torch.save(model.state_dict(), f'model_states/{args.model}.pt')
            best_loss = val_loss
        print(f"\tVal: Loss: {val_loss}")
    plot_loss(loss_history)

    # === Test single image ===
    best_state = torch.load(f'model_states/{args.model}.pt', map_location=torch.device('cpu'))
    model.load_state_dict(best_state)
    model.eval()

    # TO PREDICT SINGLE SAMPLE AND SHOW THE RESULT
    sample_id = 64
    test_image = test_data[sample_id][0].unsqueeze(0).to(device)
    test_image_pred = model(test_image)
    if isinstance(test_image_pred, OrderedDict):
        test_image_pred = test_image_pred['out']
    test_image_pred = test_image_pred.detach().cpu().squeeze(0)
    test_image_mask_values = torch.argmax(test_image_pred, dim=0)
    test_image_mask = test_image_mask_values.numpy()
    pred_mask = decode_segmap(test_image_mask)

    real_image_mask = test_data[sample_id][1].squeeze(0).numpy()
    real_image_mask_values = np.argmax(real_image_mask, axis=0)
    target_mask = decode_segmap(real_image_mask_values)

    compare_masks(test_image.squeeze().permute(1, 2, 0), target_mask, pred_mask)

    # === Run test dataset ===
    test_loss, mean_iou = validate(device, model, criterion, test_loader, iou)
    print(f"TEST: Loss: {test_loss} | Mean IoU: {mean_iou}")


if __name__ == "__main__":
    main()
