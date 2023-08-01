import os
import cv2
import torch
from torch.utils.data import Dataset
from utils import label_colors_tensor


class SUIMDataset(Dataset):
    def __init__(self, path, transforms=None, mask_t=None):
        self.dir_path = path
        self.transforms = transforms
        self.mask_t = mask_t
        self.images_names = []
        self.masks_names = []

        images_path = os.path.join(path, "images")
        masks_path = os.path.join(path, "masks")

        for image in os.listdir(images_path):
            self.images_names.append(image)
        for mask in os.listdir(masks_path):
            _, ext = os.path.splitext(mask)
            if ext == ".bmp":
                self.masks_names.append(mask)

    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dir_path, "images", self.images_names[idx])
        mask_path = os.path.join(self.dir_path, "masks", self.masks_names[idx])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            img = self.transforms(img)
            mask = self.mask_t(mask)

            # Create one-hot encoded tensor
            expanded_rgb_image = mask.unsqueeze(0)
            one_hot_encoded = torch.stack([torch.all(expanded_rgb_image == color.reshape(1, 1, 3, 1, 1), dim=2) for color in label_colors_tensor], dim=0).float()
            mask = one_hot_encoded.squeeze(1).squeeze(1)

        return img, mask
