"""
Evaluation Metrics Protocol for Saree Design Recognition.
Calculates Rank-1, Rank-5, mAP (Identification) and ROC-AUC, EER, TAR@FAR (Verification).
"""

import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def compute_cosine_similarity(embeddings_a: np.ndarray, embeddings_b: np.ndarray) -> np.ndarray:
    """
    Computes pairwise cosine similarity between two sets of L2-normalized embeddings.
    """
    norm_a = embeddings_a / (np.linalg.norm(embeddings_a, axis=1, keepdims=True) + 1e-8)
    norm_b = embeddings_b / (np.linalg.norm(embeddings_b, axis=1, keepdims=True) + 1e-8)
    return np.dot(norm_a, norm_b.T)


def evaluate_identification(query_embeddings: np.ndarray, query_labels: np.ndarray,
                            gallery_embeddings: np.ndarray, gallery_labels: np.ndarray,
                            top_k=(1, 5)):
    """
    Evaluates Gallery vs Query Identification performance.
    
    Returns:
        results (dict): Rank-1, Rank-5 accuracy, and mAP score.
    """
    sim_matrix = compute_cosine_similarity(query_embeddings, gallery_embeddings)
    num_queries = query_embeddings.shape[0]

    rank1_count = 0
    rank5_count = 0
    aps = []

    for i in range(num_queries):
        q_label = query_labels[i]
        scores = sim_matrix[i]
        
        # Sort gallery indices by descending similarity
        sorted_indices = np.argsort(-scores)
        retrieved_labels = gallery_labels[sorted_indices]

        # Check Rank-1
        if retrieved_labels[0] == q_label:
            rank1_count += 1

        # Check Rank-5
        if q_label in retrieved_labels[:5]:
            rank5_count += 1

        # Calculate Average Precision (AP) for query i
        binary_matches = (retrieved_labels == q_label).astype(int)
        num_positives = np.sum(binary_matches)
        
        if num_positives > 0:
            cum_positives = np.cumsum(binary_matches)
            precisions = cum_positives / (np.arange(len(binary_matches)) + 1)
            ap = np.sum(precisions * binary_matches) / num_positives
            aps.append(ap)
        else:
            aps.append(0.0)

    rank1_acc = rank1_count / num_queries
    rank5_acc = rank5_count / num_queries
    mAP = float(np.mean(aps))

    return {
        "Rank-1 Accuracy": rank1_acc,
        "Rank-5 Accuracy": rank5_acc,
        "mAP": mAP
    }


def evaluate_verification(embeddings_1: np.ndarray, embeddings_2: np.ndarray, labels_same: np.ndarray):
    """
    Evaluates Pairwise Verification performance.
    
    Args:
        embeddings_1 (np.ndarray): First set of pair embeddings.
        embeddings_2 (np.ndarray): Second set of pair embeddings.
        labels_same (np.ndarray): Binary targets (1 if same design, 0 if different).
        
    Returns:
        results (dict): ROC-AUC, EER, and TAR@FAR=0.01.
    """
    # Cosine similarity for each pair
    norm_1 = embeddings_1 / (np.linalg.norm(embeddings_1, axis=1, keepdims=True) + 1e-8)
    norm_2 = embeddings_2 / (np.linalg.norm(embeddings_2, axis=1, keepdims=True) + 1e-8)
    similarities = np.sum(norm_1 * norm_2, axis=1)

    fpr, tpr, thresholds = roc_curve(labels_same, similarities)
    roc_auc = auc(fpr, tpr)

    # Calculate Equal Error Rate (EER)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)

    # Calculate TAR @ FAR = 0.01
    target_far = 0.01
    far_idx = np.where(fpr <= target_far)[0]
    tar_at_far01 = float(tpr[far_idx[-1]]) if len(far_idx) > 0 else 0.0

    return {
        "ROC-AUC": roc_auc,
        "EER": eer,
        "TAR@FAR=0.01": tar_at_far01,
        "Optimal_Threshold": float(thresholds[eer_idx])
    }
