"""
End-to-End PyTorch Training Loop for Color-Invariant Saree Recognition.
DeepLure AIE-CASE Training Engine.
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import SyntheticSareeDataset, RealSareeDataset, PKSampler
from src.augmentations import get_train_transforms, get_eval_transforms
from src.models import ColorInvariantSareeNet
from src.loss import ColorInvariantCombinedLoss
from src.metrics import evaluate_identification, evaluate_verification
from src.utils import set_seed, get_model_efficiency_summary


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    loss_stats = {"arcface_loss": 0.0, "supcon_loss": 0.0}
    correct_predictions = 0
    total_samples = 0

    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        
        # Forward pass
        embeddings = model(images)
        loss, logits, batch_stats = criterion(embeddings, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        for k, v in batch_stats.items():
            loss_stats[k] += v * images.size(0)

        preds = torch.argmax(logits, dim=1)
        correct_predictions += torch.sum(preds == labels).item()
        total_samples += images.size(0)

    elapsed = time.time() - start_time
    avg_loss = total_loss / total_samples
    acc = correct_predictions / total_samples

    print(f"Epoch {epoch:02d} | Train Loss: {avg_loss:.4f} | Train Acc: {acc*100:.2f}% | Time: {elapsed:.2f}s")
    return avg_loss, acc


@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    all_embeddings = []
    all_labels = []

    for batch in val_loader:
        images = batch["image"].to(device)
        labels = batch["label"]

        embeddings = model(images)
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(labels.numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Split into Gallery (50%) and Query (50%)
    num_samples = len(all_labels)
    split_idx = num_samples // 2

    query_embeddings, gallery_embeddings = all_embeddings[:split_idx], all_embeddings[split_idx:]
    query_labels, gallery_labels = all_labels[:split_idx], all_labels[split_idx:]

    id_results = evaluate_identification(query_embeddings, query_labels, gallery_embeddings, gallery_labels)

    # Generate pairwise verification evaluation
    min_len = min(len(query_embeddings), len(gallery_embeddings))
    emb1 = query_embeddings[:min_len]
    emb2 = gallery_embeddings[:min_len]
    same_labels = (query_labels[:min_len] == gallery_labels[:min_len]).astype(int)

    verif_results = evaluate_verification(emb1, emb2, same_labels)

    print(f"Validation | Rank-1: {id_results['Rank-1 Accuracy']*100:.2f}% | Rank-5: {id_results['Rank-5 Accuracy']*100:.2f}% | mAP: {id_results['mAP']:.4f} | ROC-AUC: {verif_results['ROC-AUC']:.4f}")

    return {**id_results, **verif_results}


def main():
    parser = argparse.ArgumentParser(description="Train Color-Invariant Saree Recognition Model")
    parser.add_argument("--backbone", type=str, default="resnet34", choices=["resnet34", "resnet50", "mobilenet_v3", "convnext_tiny"])
    parser.add_argument("--embedding_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--data_dir", type=str, default=None, help="Path to real saree dataset directory")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transforms
    train_transforms = get_train_transforms(image_size=224, color_invariant_mode="heavy_jitter")
    val_transforms = get_eval_transforms(image_size=224, color_invariant_mode="heavy_jitter")

    # Dataset setup
    if args.data_dir and os.path.exists(args.data_dir):
        print(f"Loading real saree dataset from: {args.data_dir}")
        train_dataset = RealSareeDataset(root_dir=args.data_dir, transform=train_transforms)
        val_dataset = RealSareeDataset(root_dir=args.data_dir, transform=val_transforms)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    else:
        print("Initializing Synthetic Saree Motif Dataset for color-invariant evaluation...")
        train_dataset = SyntheticSareeDataset(num_classes=args.num_classes, colorways_per_class=8, samples_per_colorway=5, transform=train_transforms)
        val_dataset = SyntheticSareeDataset(num_classes=args.num_classes, colorways_per_class=4, samples_per_colorway=2, transform=val_transforms)
        
        sampler = PKSampler(train_dataset, p=8, k=4)
        train_loader = DataLoader(train_dataset, sampler=sampler, batch_size=32)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model
    model = ColorInvariantSareeNet(backbone_name=args.backbone, embedding_size=args.embedding_size, pretrained=True).to(device)
    
    # Loss & Optimizer
    criterion = ColorInvariantCombinedLoss(in_features=args.embedding_size, num_classes=args.num_classes, s=30.0, m=0.50).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Efficiency Summary
    get_model_efficiency_summary(model, input_size=(1, 3, 224, 224))

    best_map = 0.0

    print("Starting training pipeline...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        metrics = validate(model, val_loader, device)
        scheduler.step()

        if metrics["mAP"] > best_map:
            best_map = metrics["mAP"]
            ckpt_path = os.path.join(args.save_dir, f"best_saree_{args.backbone}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics
            }, ckpt_path)
            print(f"--> Saved best model checkpoint to {ckpt_path} (mAP: {best_map:.4f})")

    print(f"\nTraining completed! Peak Validation mAP: {best_map:.4f}")


if __name__ == "__main__":
    main()
