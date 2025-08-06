import os
import torch
import random
from PIL import Image
from torch.utils.data import Dataset

from params import COLOR_TO_CLASS, TEST_PCT

SEED_FOR_TEST_SET_REPRODUCIBILITY = 42

class TomatoDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None, mode='train'):
        assert mode in ['train', 'test'], "mode must be 'train' or 'test'"
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.test_pct = TEST_PCT
        self.seed = SEED_FOR_TEST_SET_REPRODUCIBILITY

        all_samples = []

        # === Gather all samples from all subfolders ===
        for subdir in sorted(os.listdir(root_dir)):
            subdir_path = os.path.join(root_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue

            label_dir = os.path.join(subdir_path, 'annotated_labels')
            if not os.path.isdir(label_dir):
                continue

            # Only collect samples with matching labels
            for filename in sorted(os.listdir(subdir_path)):
                if filename.endswith('.png') and filename != 'annotated_labels':
                    image_path = os.path.join(subdir_path, filename)
                    label_path = os.path.join(label_dir, filename)
                    if os.path.exists(label_path):
                        all_samples.append((image_path, label_path))

        # === Deterministic shuffle and split ===
        random.seed(SEED_FOR_TEST_SET_REPRODUCIBILITY)
        all_samples.sort()  # Ensure consistent order
        random.shuffle(all_samples)

        test_size = int(len(all_samples) * TEST_PCT)
        self.test_samples = all_samples[:test_size]
        self.train_samples = all_samples[test_size:]

        self.samples = self.test_samples if self.mode == 'test' else self.train_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('RGB')  # Still RGB for color-to-class mapping

        if self.transform:
            image = self.transform(image)

        label_tensor = self._convert_label_to_class_tensor(label)
        return image, label_tensor

    def _convert_label_to_class_tensor(self, label_img):
        label_np = torch.ByteTensor(torch.ByteStorage.from_buffer(label_img.tobytes()))
        label_np = label_np.view(label_img.size[1], label_img.size[0], 3)  # H, W, C
        class_mask = torch.zeros((label_img.size[1], label_img.size[0]), dtype=torch.long)

        for color, class_id in COLOR_TO_CLASS.items():
            match = (label_np == torch.tensor(color, dtype=torch.uint8)).all(dim=-1)
            class_mask[match] = class_id

        return class_mask
