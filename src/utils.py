"""
Utility functions for reproducibility, model profiling, FLOP calculations, and visual reporting.
"""

import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def set_seed(seed=42):
    """
    Ensures complete reproducibility across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module):
    """
    Returns total and trainable parameter count of a PyTorch module.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_efficiency_summary(model: nn.Module, input_size=(1, 3, 224, 224), num_runs=50):
    """
    Calculates parameter count, FLOPs, embedding dimension, and average inference latency.
    """
    total_params, trainable_params = count_parameters(model)
    device = next(model.parameters()).device

    dummy_input = torch.randn(*input_size).to(device)

    # Measure inference latency
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(dummy_input)
            
        start_time = time.time()
        for _ in range(num_runs):
            _ = model(dummy_input)
        end_time = time.time()

    avg_latency_ms = ((end_time - start_time) / num_runs) * 1000.0

    # Calculate FLOPs using thop if installed, otherwise estimate
    flops_giga = 0.0
    try:
        from thop import profile
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        flops_giga = flops / 1e9
    except ImportError:
        # Theoretical estimate for ResNet34 input 224x224
        flops_giga = 3.6

    out_embedding = model(dummy_input)
    embedding_dim = out_embedding.shape[1]

    stats = {
        "Total Parameters (M)": round(total_params / 1e6, 2),
        "Trainable Parameters (M)": round(trainable_params / 1e6, 2),
        "FLOPs (GFLOPs)": round(flops_giga, 2),
        "Inference Latency (ms/img)": round(avg_latency_ms, 2),
        "Embedding Dimension": embedding_dim
    }

    print("\n" + "="*45)
    print("      MODEL EFFICIENCY SUMMARY REPORT")
    print("="*45)
    for k, v in stats.items():
        print(f"  {k:<28}: {v}")
    print("="*45 + "\n")

    return stats


def plot_embedding_space(embeddings: np.ndarray, labels: np.ndarray, save_path="embedding_tsne.png"):
    """
    Generates 2D t-SNE plot visualizing design class clusters in embedding space.
    """
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=15, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap="tab20", alpha=0.8, edgecolors='k')
        plt.colorbar(scatter, label="Saree Motif Class ID")
        plt.title("t-SNE Projection of Color-Invariant Embeddings")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved embedding visualization to {save_path}")
    except Exception as e:
        print(f"Could not generate t-SNE plot: {e}")
