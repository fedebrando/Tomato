
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter 

from model import UNET
from dataset import TomatoDataset
from metrics import acc_metric
from params import CLASS_TO_COLOR
from sklearn.metrics import confusion_matrix  
from arg_parser import get_test_args

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


def plot_sample_grid(x, pred_mask, target_mask, idx, save_dir):
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

# Funzione per calcolare le metriche
def calculate_metrics(pred_mask, true_mask, num_classes=NUM_CLASSES):
    """Calcola le metriche di segmentazione: IoU, Precision, Recall, F1, Dice"""
    
    # Flatten le maschere per confrontare tutti i pixel
    pred_flat = pred_mask.flatten()
    true_flat = true_mask.flatten()

    # Calcola la matrice di confusione
    cm = confusion_matrix(true_flat, pred_flat, labels=np.arange(num_classes))

    # Calcolare le metriche per ogni classe
    iou_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    dice_list = []
    accuracy = torch.sum(pred_flat == true_flat) / len(true_flat)  # Accuracy globale

    for i in range(num_classes):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP
        TN = np.sum(cm) - TP - FP - FN
        
        # IoU
        iou = TP / (TP + FP + FN) if (TP + FP + FN) != 0 else 0
        iou_list.append(iou)
        
        # Precision
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        precision_list.append(precision)
        
        # Recall
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        recall_list.append(recall)
        
        # F1-score
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        f1_list.append(f1)
        
        # Dice coefficient
        dice = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0
        dice_list.append(dice)
    
    # Media delle metriche per ogni classe
    mean_iou = np.mean(iou_list)
    mean_precision = np.mean(precision_list)
    mean_recall = np.mean(recall_list)
    mean_f1 = np.mean(f1_list)
    mean_dice = np.mean(dice_list)
    
    return {
        "accuracy": accuracy,
        "mean_iou": mean_iou,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "mean_f1": mean_f1,
        "mean_dice": mean_dice,
        "iou_per_class": iou_list,
        "precision_per_class": precision_list,
        "recall_per_class": recall_list,
        "f1_per_class": f1_list,
        "dice_per_class": dice_list
    }

def main(args):
    # === CONFIGURATION ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join('..', 'models', args.model_name, 'best_model.pth')
    data_root = 'images'
    batch_size = 4

    # === CREATE A NEW DIRECTORY FOR TENSORBOARD LOGS ===
    log_dir = os.path.join("..", "runs", args.model_name, "test")  # Create a directory based on the model name
    writer = SummaryWriter(log_dir)  # Initialize TensorBoard writer

    # === LOAD MODEL ===
    model = UNET(3, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval().to(device)

    # === DATASET ===
    transform = transforms.Compose([transforms.ToTensor()])
    test_set = TomatoDataset(data_root, mode='test')

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # === EVALUATION ===
    total_metrics = {
        "accuracy": 0.0,
        "mean_iou": 0.0,
        "mean_precision": 0.0,
        "mean_recall": 0.0,
        "mean_f1": 0.0,
        "mean_dice": 0.0
    }
    
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Evaluating"):
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            pred_mask = preds.argmax(dim=1)  # Assicurati che la previsione abbia la forma giusta

            # Calcola le metriche per la segmentazione
            metrics = calculate_metrics(pred_mask.cpu(), y.cpu())  # Passa le maschere sulla CPU per calcolare le metriche

            for key in total_metrics:
                total_metrics[key] += metrics[key]

    # Final metrics
    num_batches = len(test_loader)
    for key in total_metrics:
        total_metrics[key] /= num_batches

    print(f"\n✅ Accuracy: {total_metrics['accuracy']:.4f}")
    print(f"✅ Mean IoU: {total_metrics['mean_iou']:.4f}")
    print(f"✅ Mean Precision: {total_metrics['mean_precision']:.4f}")
    print(f"✅ Mean Recall: {total_metrics['mean_recall']:.4f}")
    print(f"✅ Mean F1: {total_metrics['mean_f1']:.4f}")
    print(f"✅ Mean Dice: {total_metrics['mean_dice']:.4f}")
    
    # Costruisci tabella markdown come stringa
    # === Tabella Markdown per TensorBoard con medie e per-classe ===
    CLASS_NAMES = ["Background", "Tomato", "Leaves", "Vase", "Floor", "Trunk"]  # oppure importala da params.py

    # === Tabella 1: Metriche medie ===
    mean_metrics_table = "| Metric | Value |\n|--------|-------|\n"
    for key in ["accuracy", "mean_iou", "mean_precision", "mean_recall", "mean_f1", "mean_dice"]:
        mean_metrics_table += f"| {key} | {metrics[key]:.4f} |\n"

    # === Tabella 2: Metriche per classe ===
    per_class_table = "| Class | IoU | Precision | Recall | F1 | Dice |\n"
    per_class_table += "|-------|-----|-----------|--------|----|------|\n"
    for i, class_name in enumerate(CLASS_NAMES):
        per_class_table += (
            f"| {class_name} | "
            f"{metrics['iou_per_class'][i]:.4f} | "
            f"{metrics['precision_per_class'][i]:.4f} | "
            f"{metrics['recall_per_class'][i]:.4f} | "
            f"{metrics['f1_per_class'][i]:.4f} | "
            f"{metrics['dice_per_class'][i]:.4f} |\n"
        )

    # === Logga le tabelle su TensorBoard ===
    writer.add_text("Metrics/Mean", mean_metrics_table, global_step=0)
    writer.add_text("Metrics/Per Class", per_class_table, global_step=0)

    # === SAVE VISUALIZATION ===
    if args.save_preds:
        model.cpu()
        print("📸 Saving visual samples...")
        
        save_dir = os.path.join("..", "runs", args.model_name, "inference")
        os.makedirs(save_dir, exist_ok=True)

        for i in range(5):
            x, y = test_set[i]
            pred = model(x.unsqueeze(0)).squeeze(0)
            pred_mask = pred.argmax(dim=0).byte()

            # Salva la predizione originale (maschera)
            pred_mask_img = Image.fromarray(pred_mask.cpu().numpy())  # Converti la maschera in immagine
            pred_mask_img.save(os.path.join(save_dir, f"pred_mask_{i}.png"))

            pred_rgb = decode_segmentation_mask(pred_mask.cpu().numpy())  # Converti la maschera in RGB
            pred_rgb.save(os.path.join(save_dir, f"pred_rgb_{i}.png"))

            # Salva anche la visualizzazione della griglia (originale + overlay)
            plot_sample_grid(x, pred_mask, y, idx=i, save_dir=save_dir)

        print(f"✅ Visual samples saved to {save_dir}")

if __name__ == "__main__":
    args = get_test_args() 
    main(args)
