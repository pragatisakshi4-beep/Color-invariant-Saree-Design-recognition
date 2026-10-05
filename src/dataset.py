"""
Saree Dataset Module & Synthetic Motif Generator.
Includes procedural synthetic motif dataset generation with multi-colorways
and PyTorch Dataset / DataLoader utilities for DeepLure & Kaggle datasets.
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, Sampler


class SyntheticSareeMotifGenerator:
    """
    Procedurally generates synthetic saree design motifs with multiple colorways.
    Used for instant self-contained testing, reproducibility, and Kaggle validation.
    """
    MOTIF_TYPES = [
        "paisley", "temple_border", "bandhani_dots", "floral_jaal",
        "geometric_lattice", "chevron_stripes", "peacock_feather", "chanderi_checks"
    ]

    COLORWAYS = [
        ((180, 20, 30), (255, 215, 0)),    # Crimson & Gold
        ((0, 100, 120), (255, 127, 80)),   # Teal & Coral
        ((75, 0, 130), (255, 165, 0)),     # Indigo & Orange
        ((34, 139, 34), (255, 255, 220)),  # Emerald & Off-white
        ((20, 20, 20), (200, 200, 200)),   # Charcoal & Silver
        ((139, 0, 0), (240, 230, 140)),    # Maroon & Khaki
        ((255, 20, 147), (0, 255, 255)),   # Deep Pink & Cyan
        ((46, 139, 87), (255, 215, 0)),    # Sea Green & Gold
    ]

    @staticmethod
    def generate_motif_image(motif_id: int, colorway_id: int, img_size=224) -> Image.Image:
        motif_type = SyntheticSareeMotifGenerator.MOTIF_TYPES[motif_id % len(SyntheticSareeMotifGenerator.MOTIF_TYPES)]
        bg_color, fg_color = SyntheticSareeMotifGenerator.COLORWAYS[colorway_id % len(SyntheticSareeMotifGenerator.COLORWAYS)]
        
        # Add slight background gradient or texture variation
        img = Image.new("RGB", (img_size, img_size), bg_color)
        draw = ImageDraw.Draw(img)
        
        grid_step = img_size // 7
        
        if motif_type == "paisley":
            for r in range(grid_step // 2, img_size, grid_step):
                for c in range(grid_step // 2, img_size, grid_step):
                    draw.ellipse([c-10, r-15, c+10, r+15], fill=fg_color)
                    draw.polygon([(c, r-20), (c-12, r), (c+12, r)], fill=fg_color)
                    
        elif motif_type == "temple_border":
            for i in range(0, img_size, 30):
                draw.polygon([(i, 50), (i+15, 10), (i+30, 50)], fill=fg_color)
                draw.polygon([(i, img_size-50), (i+15, img_size-10), (i+30, img_size-50)], fill=fg_color)
            draw.line([(0, 55), (img_size, 55)], fill=fg_color, width=4)
            draw.line([(0, img_size-55), (img_size, img_size-55)], fill=fg_color, width=4)
            
        elif motif_type == "bandhani_dots":
            for r in range(15, img_size, 25):
                for c in range(15, img_size, 25):
                    draw.ellipse([c-5, r-5, c+5, r+5], fill=fg_color, outline=bg_color)
                    
        elif motif_type == "floral_jaal":
            for r in range(25, img_size, 45):
                for c in range(25, img_size, 45):
                    draw.line([(c-20, r), (c+20, r)], fill=fg_color, width=2)
                    draw.line([(c, r-20), (c, r+20)], fill=fg_color, width=2)
                    draw.ellipse([c-8, r-8, c+8, r+8], fill=fg_color)
                    
        elif motif_type == "geometric_lattice":
            for i in range(-img_size, img_size * 2, 25):
                draw.line([(i, 0), (i + img_size, img_size)], fill=fg_color, width=2)
                draw.line([(i, img_size), (i + img_size, 0)], fill=fg_color, width=2)
                
        elif motif_type == "chevron_stripes":
            for y in range(0, img_size, 30):
                points = [(0, y), (img_size//2, y+15), (img_size, y)]
                draw.line(points, fill=fg_color, width=5)
                
        elif motif_type == "peacock_feather":
            for r in range(40, img_size, 60):
                for c in range(40, img_size, 60):
                    draw.ellipse([c-15, r-25, c+15, r+25], fill=fg_color)
                    draw.ellipse([c-7, r-12, c+7, r+12], fill=bg_color)
                    
        elif motif_type == "chanderi_checks":
            for i in range(0, img_size, 20):
                draw.line([(i, 0), (i, img_size)], fill=fg_color, width=3)
                draw.line([(0, i), (img_size, i)], fill=fg_color, width=3)
                
        return img


class SyntheticSareeDataset(Dataset):
    """
    Dataset of procedurally generated saree motifs across different colorways.
    """
    def __init__(self, num_classes=20, colorways_per_class=8, samples_per_colorway=5, transform=None, img_size=224):
        self.num_classes = num_classes
        self.colorways_per_class = colorways_per_class
        self.samples_per_colorway = samples_per_colorway
        self.transform = transform
        self.img_size = img_size
        
        self.data = []
        for class_id in range(num_classes):
            for colorway_id in range(colorways_per_class):
                for s in range(samples_per_colorway):
                    self.data.append({
                        "class_id": class_id,
                        "colorway_id": colorway_id,
                        "sample_id": s
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        class_id = item["class_id"]
        colorway_id = item["colorway_id"]
        
        # Generate motif image with reproducible seed based on idx
        img = SyntheticSareeMotifGenerator.generate_motif_image(
            motif_id=class_id,
            colorway_id=colorway_id,
            img_size=self.img_size
        )
        
        if self.transform:
            img = self.transform(img)
            
        return {
            "image": img,
            "label": torch.tensor(class_id, dtype=torch.long),
            "colorway": torch.tensor(colorway_id, dtype=torch.long)
        }


class RealSareeDataset(Dataset):
    """
    PyTorch Dataset for loading images from disk (DeepLure or Kaggle saree datasets).
    Supports folder structure: root/class_name/image.jpg
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        
        if os.path.exists(root_dir):
            classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
            
            for cls_name in classes:
                cls_dir = os.path.join(root_dir, cls_name)
                cls_id = self.class_to_idx[cls_name]
                for fname in os.listdir(cls_dir):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                        self.samples.append((os.path.join(cls_dir, fname), cls_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "path": path
        }


class PKSampler(Sampler):
    """
    P-K Sampler for Metric Learning:
    Samples P random design classes, and K images per class in each batch.
    Ensures that every mini-batch contains positive colorway pairs for contrastive loss.
    """
    def __init__(self, dataset, p=8, k=4):
        self.dataset = dataset
        self.p = p
        self.k = k
        self.batch_size = p * k
        
        # Group indices by class label
        self.label_to_indices = {}
        for idx in range(len(dataset)):
            item = dataset[idx]
            label = item["label"].item() if isinstance(item["label"], torch.Tensor) else item["label"]
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
            
        self.labels = list(self.label_to_indices.keys())

    def __iter__(self):
        labels_copy = self.labels.copy()
        random.shuffle(labels_copy)
        
        batches = []
        while len(labels_copy) >= self.p:
            batch_labels = labels_copy[:self.p]
            labels_copy = labels_copy[self.p:]
            
            batch = []
            for label in batch_labels:
                indices = self.label_to_indices[label]
                if len(indices) >= self.k:
                    selected = random.sample(indices, self.k)
                else:
                    selected = np.random.choice(indices, self.k, replace=True).tolist()
                batch.extend(selected)
            batches.extend(batch)
            
        return iter(batches)

    def __len__(self):
        return (len(self.labels) // self.p) * self.batch_size
