
import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class TomatoDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.samples = []

        # Scansione delle sottocartelle
        for subdir in sorted(os.listdir(root_dir)):
            subdir_path = os.path.join(root_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue

            label_dir = os.path.join(subdir_path, 'annotated_labels')
            if not os.path.isdir(label_dir):
                continue

            for filename in sorted(os.listdir(subdir_path)):
                if filename.endswith('.png') and filename != 'annotated_labels':
                    image_path = os.path.join(subdir_path, filename)
                    label_path = os.path.join(label_dir, filename)
                    if os.path.exists(label_path):
                        self.samples.append((image_path, label_path))

        # Mappa colori → classi
        self.color_to_class = {
            (255, 0, 0): 1,       # red (tomato)
            (0, 255, 0): 2,       # green (leaf)
            (0, 0, 255): 3,       # blue (vase)
            (125, 125, 0): 4,     # (floor)
            (255, 255, 0): 5      # (trunk)
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('RGB')  # resta RGB per mappare i colori

        if self.transform:
            image = self.transform(image)

        label_tensor = self._convert_label_to_class_tensor(label)

        return image, label_tensor

    def _convert_label_to_class_tensor(self, label_img):
        label_np = torch.ByteTensor(torch.ByteStorage.from_buffer(label_img.tobytes()))
        label_np = label_np.view(label_img.size[1], label_img.size[0], 3)  # H, W, C
        class_mask = torch.zeros((label_img.size[1], label_img.size[0]), dtype=torch.long)

        for color, class_id in self.color_to_class.items():
            match = (label_np == torch.tensor(color, dtype=torch.uint8)).all(dim=-1)
            class_mask[match] = class_id

        return class_mask
