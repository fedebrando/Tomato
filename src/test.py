
from dataset import *
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import functional as F

transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = TomatoDataset('../images', transform=transform, target_transform=transform)

image, label = dataset[0]
print(image.shape, label.shape)

# Convertili per la visualizzazione (da tensor a numpy)
image_np = image.permute(1, 2, 0).numpy()  # [C,H,W] → [H,W,C]
label_np = label.numpy()         # Rimuove il canale per grayscale o maschera

# Visualizza immagine e ground truth
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Input Image')
plt.imshow(image.permute(1, 2, 0))  # [C,H,W] → [H,W,C]
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Ground Truth (Class Map)')
plt.imshow(label, cmap='gray', vmin=0, vmax=5)
plt.axis('off')

plt.tight_layout()
plt.show()

