
import os
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
 
from params import COLOR_TO_CLASS, TEST_PCT
 
SEED_FOR_TEST_SET_REPRODUCIBILITY = 42
 
class SegmentationAugmentation:
    def __init__(self, flip_prob, rotate_prob, color_jitter_prob, crop_prob):
        self.flip_prob = flip_prob # 0.5
        self.rotate_prob = rotate_prob # 0.4
        self.color_jitter_prob = color_jitter_prob # 0.5
        self.crop_prob = crop_prob # 0.4
        
        self.rotation_degrees = 30
        self.color_jitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
        self.crop_size = (256, 256)
        self.resize_back = (480, 848)

    def __call__(self, img, mask):
        if random.random() < self.flip_prob:
            img = F.hflip(img)
            mask = F.hflip(mask)

        if random.random() < self.rotate_prob:
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            img = F.rotate(img, angle)
            mask = F.rotate(mask, angle)

        if random.random() < self.color_jitter_prob:
            img = self.color_jitter(img)

        if random.random() < self.crop_prob:
            i, j, h, w = transforms.RandomCrop.get_params(img, output_size=self.crop_size)
            img = F.crop(img, i, j, h, w)
            mask = F.crop(mask, i, j, h, w)

            img = F.resize(img, self.resize_back)
            mask = F.resize(mask, self.resize_back)

        img = F.to_tensor(img)
        return img, mask
 
class TomatoDataset(Dataset):
    def __init__(self, root_dir, flip_prob=0, rotate_prob=0, jitter_prob=0, crop_prob=0, mode='train'):
        assert mode in ['train', 'test'], "mode must be 'train' or 'test'"
        self.root_dir = root_dir
        self.aug = any([flip_prob, rotate_prob, jitter_prob, crop_prob])
        self.mode = mode
        if self.aug and self.mode == 'train':
            self.transform = SegmentationAugmentation(flip_prob, rotate_prob, jitter_prob, crop_prob)
        else:
            self.transform = transforms.ToTensor()
 
        all_samples = []
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
                        all_samples.append((image_path, label_path))
 
        random.seed(SEED_FOR_TEST_SET_REPRODUCIBILITY)
        all_samples.sort()
        random.shuffle(all_samples)
 
        test_size = int(len(all_samples) * TEST_PCT)
        self.test_samples = all_samples[:test_size]
        self.train_samples = all_samples[test_size:]
        self.samples = self.test_samples if mode == 'test' else self.train_samples
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('RGB')
 
        if self.aug:
            image, label = self.transform(image, label)
        else:
            image = self.transform(image)
        # Ora converte la maschera in classi
        label_tensor = self._convert_label_to_class_tensor(label)
 
        if not torch.is_tensor(image):
            image = F.to_tensor(image)
 
        return image, label_tensor
 
    def _convert_label_to_class_tensor(self, label_img):
        if torch.is_tensor(label_img):
            label_img = Image.fromarray(label_img.numpy().astype(np.uint8))
 
        label_np = torch.ByteTensor(torch.ByteStorage.from_buffer(label_img.tobytes()))
        label_np = label_np.view(label_img.size[1], label_img.size[0], 3)  # H, W, C
        class_mask = torch.zeros((label_img.size[1], label_img.size[0]), dtype=torch.long)
 
        for color, class_id in COLOR_TO_CLASS.items():
            match = (label_np == torch.tensor(color, dtype=torch.uint8)).all(dim=-1)
            class_mask[match] = class_id
 
        return class_mask
