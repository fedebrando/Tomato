import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

from model import UNET
from dataset import TomatoDataset
from metrics import acc_metric
from params import CLASS_TO_COLOR

NUM_CLASSES = 6

def decode_segmentation_mask(mask):
    """Convert class ID mask to RGB mask."""
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_TO_COLOR.items():
        rgb_mask[mask == class_id] = color
    return Image.fromarray(rgb_mask)


def create_overlay(input_tensor, mask_tensor):
    """Overlay a color mask over an input image with transparency."""
    input_img = input_tensor.permute(1, 2, 0).numpy()
    input_img = (input_img * 255).astype(np.uint8)
    base = Image.fromarray(input_img).convert("RGBA")

    color_mask = decode_segmentation_mask(mask_tensor.numpy()).convert("RGBA")
    mask_np = np.array(color_mask)
    mask_np[mask_tensor.numpy() == 0, 3] = 0  # make background transparent
    mask_overlay = Image.fromarray(mask_np)

    return Image.alpha_composite(base, mask_overlay)


def plot_sample_grid(x, pred_mask, target_mask, idx, save_dir="../predictions"):
    """
    Save a 3-row, 2-column figure:
        Row 1: Input only
        Row 2: Prediction, Prediction Overlay
        Row 3: Ground Truth, GT Overlay
    """
    os.makedirs(save_dir, exist_ok=True)

    input_img = x.permute(1, 2, 0).numpy()
    input_img = (input_img * 255).astype(np.uint8)

    pred_rgb = decode_segmentation_mask(pred_mask.numpy())
    gt_rgb = decode_segmentation_mask(target_mask.numpy())

    pred_overlay = create_overlay(x, pred_mask)
    gt_overlay = create_overlay(x, target_mask)

    # Create the grid: 3 rows x 2 columns
    fig, axes = plt.subplots(3, 2, figsize=(8, 10))

    # Row 1: input + empty cell
    axes[0, 0].imshow(input_img)
    axes[0, 0].set_title("Input")
    axes[0, 1].axis('off')  # empty cell

    # Row 2: prediction + overlay
    axes[1, 0].imshow(pred_rgb)
    axes[1, 0].set_title("Prediction")
    axes[1, 1].imshow(pred_overlay)
    axes[1, 1].set_title("Prediction Overlay")

    # Row 3: ground truth + overlay
    axes[2, 0].imshow(gt_rgb)
    axes[2, 0].set_title("Ground Truth")
    axes[2, 1].imshow(gt_overlay)
    axes[2, 1].set_title("GT Overlay")

    # Remove axes
    for row in axes:
        for ax in row:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"sample_{idx}.png"))
    plt.close()


def main():
    # === CONFIGURATION ===
    model_path = os.path.join('..', 'models', 'best_model.pth')
    data_root = os.path.join('..', 'images')
    batch_size = 4
    save_preds = True

    # === LOAD MODEL ===
    model = UNET(3, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path))
    model.eval().cuda()

    # === DATASET ===
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = TomatoDataset(data_root, transform=transform, target_transform=None)

    total_len = len(dataset)
    train_len = int(total_len * 0.7)
    val_len = int(total_len * 0.15)
    test_len = total_len - train_len - val_len

    _, _, test_dataset = torch.utils.data.random_split(dataset, [train_len, val_len, test_len])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # === EVALUATION ===
    total_acc = 0.0
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Evaluating"):
            x = x.cuda()
            y = y.cuda()
            preds = model(x)
            acc = acc_metric(preds, y)
            total_acc += acc.item()

    avg_acc = total_acc / len(test_loader)
    print(f"\n✅ Accuracy on test set: {avg_acc:.4f}")

    # === SAVE VISUALIZATION ===
    if save_preds:
        model.cpu()
        print("📸 Saving visual samples...")
        for i in range(5):
            x, y = test_dataset[i]
            pred = model(x.unsqueeze(0)).squeeze(0)
            pred_mask = pred.argmax(dim=0).byte()
            plot_sample_grid(x, pred_mask, y, idx=i)
        print("✅ Visual samples saved to ../predictions/")

if __name__ == "__main__":
    main()
