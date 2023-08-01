import numpy as np
import torch
import matplotlib.pyplot as plt


label_colors = np.array([(0, 0, 0),         # 0=Background waterbody
                        (255, 0, 0),       # 1=Robots/instruments
                        (0, 255, 0),       # 2=Plants/sea-grass
                        (0, 0, 255),       # 3=Human divers
                        (255, 255, 0),     # 4=Fish and vertebrates
                        (255, 0, 255),     # 5=Reefs and invertebrates
                        (0, 255, 255),     # 6=Wrecks/ruins
                        (255, 255, 255)])  # 7=Sand/sea-floor (& rocks)

label_colors_tensor = torch.tensor(label_colors, dtype=torch.uint8)


def decode_segmap(image, nc=8):
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb


def plot_loss(loss_history: dict):
    epochs = len(loss_history["train"])
    plt.plot(range(epochs), loss_history["train"], label="train")
    plt.plot(range(epochs), loss_history["val"], label="val")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def compare_masks(image, target, pred):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_facecolor("#e1ddbf")
    ax1.imshow(image)
    ax1.set_title("Image")
    ax1.axis('off')
    ax2.imshow(target)
    ax2.set_title("Target")
    ax2.axis('off')
    ax3.imshow(pred)
    ax3.set_title("Prediction")
    ax3.axis('off')
    plt.show()
