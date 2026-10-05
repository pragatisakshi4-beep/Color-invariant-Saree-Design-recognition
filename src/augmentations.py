"""
Color-Invariant Augmentations Pipeline for Saree Design Recognition.
Strips chromatic bias while preserving fine structural motif features.
"""

import math
import random
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter, ImageOps
import numpy as np


class ColorPermutationTransform(object):
    """
    Randomly permutes or replaces color palettes to enforce color invariance.
    Swaps RGB channels or maps colors into randomized synthetic palettes.
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        
        mode = random.choice(["permute_channels", "palette_swap", "solarize", "posterize"])
        img_np = np.array(img)
        
        if mode == "permute_channels":
            channels = [0, 1, 2]
            random.shuffle(channels)
            img_np = img_np[:, :, channels]
            return Image.fromarray(img_np)
            
        elif mode == "palette_swap":
            # Invert colors or shift hue aggressively
            hue_shift = random.randint(0, 180)
            img_hsv = cv2_hsv_shift(img_np, hue_shift) if 'cv2_hsv_shift' in globals() else img_np
            return Image.fromarray(img_hsv)
            
        elif mode == "solarize":
            threshold = random.randint(64, 192)
            return ImageOps.solarize(img, threshold)
            
        elif mode == "posterize":
            bits = random.randint(2, 5)
            return ImageOps.posterize(img, bits)
            
        return img


def cv2_hsv_shift(img_np: np.ndarray, shift: int) -> np.ndarray:
    try:
        import cv2
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + shift) % 180
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    except ImportError:
        return img_np


class LABLChannelExtraction(object):
    """
    Converts RGB image into LAB color space and replaces RGB channels with L (Lightness),
    plus Sobel edge features for structure preservation.
    """
    def __init__(self, use_edges=True):
        self.use_edges = use_edges

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img)
        try:
            import cv2
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0]
            
            if self.use_edges:
                # Calculate Sobel edge magnitude
                sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=3)
                edge_mag = cv2.magnitude(sobelx, sobely)
                edge_mag = np.clip(edge_mag, 0, 255).astype(np.uint8)
                
                # Combine L-channel, Edge Map, and Grayscale
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                combined = np.stack([l_channel, edge_mag, gray], axis=-1)
                return Image.fromarray(combined)
            else:
                return Image.fromarray(np.stack([l_channel]*3, axis=-1))
        except ImportError:
            # Fallback to PIL Grayscale
            gray = ImageOps.grayscale(img)
            return Image.merge("RGB", (gray, gray, gray))


def get_train_transforms(image_size=224, color_invariant_mode="heavy_jitter"):
    """
    Constructs PyTorch data augmentation pipeline tailored for textile design recognition.
    
    Args:
        image_size (int): Target resolution (default 224x224).
        color_invariant_mode (str): Mode for color invariance ('heavy_jitter', 'lab_sobel', or 'grayscale').
    """
    transforms_list = [
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.3),
        T.RandomRotation(degrees=15),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    ]

    if color_invariant_mode == "heavy_jitter":
        transforms_list.extend([
            ColorPermutationTransform(p=0.6),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.5, hue=0.5),
            T.RandomGrayscale(p=0.4),
        ])
    elif color_invariant_mode == "lab_sobel":
        transforms_list.append(LABLChannelExtraction(use_edges=True))
    elif color_invariant_mode == "grayscale":
        transforms_list.append(T.Grayscale(num_output_channels=3))

    transforms_list.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3))
    ])

    return T.Compose(transforms_list)


def get_eval_transforms(image_size=224, color_invariant_mode="heavy_jitter"):
    """
    Evaluation transforms (Deterministic).
    """
    transforms_list = [
        T.Resize((image_size, image_size)),
    ]

    if color_invariant_mode == "lab_sobel":
        transforms_list.append(LABLChannelExtraction(use_edges=True))
    elif color_invariant_mode == "grayscale":
        transforms_list.append(T.Grayscale(num_output_channels=3))

    transforms_list.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return T.Compose(transforms_list)
