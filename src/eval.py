"""
Evaluation Runner for Color-Invariant Saree Recognition.
Generates full benchmark reports for Identification and Verification.
"""

import os
import argparse
import json
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.dataset import SyntheticSareeDataset, RealSareeDataset
from src.augmentations import get_eval_transforms
from src.models import ColorInvariantSareeNet
from src.metrics import evaluate_identification, evaluate_verification
from src.utils import set_seed, get_model_efficiency_summary


@torch.no_grad()
def evaluate_full_protocol(model, dataset, device, batch_size=32):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embeddings = []
    all_labels = []

    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["label"]
        
        embeddings = model(images)
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(labels.numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 50/50 Split into Query and Gallery sets
    num_samples = len(all_labels)
    split_idx = num_samples // 2

    query_emb, gallery_emb = all_embeddings[:split_idx], all_embeddings[split_idx:]
    query_lbl, gallery_lbl = all_labels[:split_idx], all_labels[split_idx:]

    print("\n" + "="*50)
    print("      EVALUATION BENCHMARK RESULTS")
    print("="*50)

    # Identification
    id_results = evaluate_identification(query_emb, query_lbl, gallery_emb, gallery_lbl)
    print("\n--- 1. Identification Protocol (Gallery vs Query Retrieval) ---")
    for k, v in id_results.items():
        if "Accuracy" in k:
            print(f"  {k:<20}: {v*100:.2f}%")
        else:
            print(f"  {k:<20}: {v:.4f}")

    # Verification
    min_len = min(len(query_emb), len(gallery_emb))
    emb1 = query_emb[:min_len]
    emb2 = gallery_emb[:min_len]
    same_labels = (query_lbl[:min_len] == gallery_lbl[:min_len]).astype(int)

    verif_results = evaluate_verification(emb1, emb2, same_labels)
    print("\n--- 2. Verification Protocol (Pairwise Matching) ---")
    for k, v in verif_results.items():
        print(f"  {k:<20}: {v:.4f}")

    full_results = {**id_results, **verif_results}
    return full_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Saree Design Recognition Model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained PyTorch model checkpoint (.pth)")
    parser.add_argument("--backbone", type=str, default="resnet34")
    parser.add_argument("--embedding_size", type=int, default=512)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--out_json", type=str, default="eval_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ColorInvariantSareeNet(backbone_name=args.backbone, embedding_size=args.embedding_size, pretrained=True).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from: {args.checkpoint}")

    val_transforms = get_eval_transforms(image_size=224, color_invariant_mode="heavy_jitter")

    if args.data_dir and os.path.exists(args.data_dir):
        eval_dataset = RealSareeDataset(root_dir=args.data_dir, transform=val_transforms)
    else:
        eval_dataset = SyntheticSareeDataset(num_classes=20, colorways_per_class=6, samples_per_colorway=4, transform=val_transforms)

    results = evaluate_full_protocol(model, eval_dataset, device)

    eff_stats = get_model_efficiency_summary(model)
    results["efficiency"] = eff_stats

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nEvaluation summary saved to {args.out_json}")


if __name__ == "__main__":
    main()
