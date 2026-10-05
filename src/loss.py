"""
Metric Learning Loss Functions for Color-Invariant Pattern Alignment.
Includes ArcFace Loss, Supervised Contrastive (SupCon) Loss, and Combined Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """
    ArcFace Angular Margin Cross Entropy Loss.
    """
    def __init__(self, in_features, num_classes, s=30.0, m=0.50):
        super(ArcFaceLoss, self).__init__()
        from src.models import ArcMarginProduct
        self.arc_head = ArcMarginProduct(in_features, num_classes, s=s, m=m)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, embeddings, labels):
        logits = self.arc_head(embeddings, labels)
        loss = self.criterion(logits, labels)
        return loss, logits


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss.
    Ref: Khosla et al. "Supervised Contrastive Learning" (NeurIPS 2020)
    Pulls embeddings of the same saree design together regardless of color palette.
    """
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = features.device

        if len(features.shape) < 2:
            raise ValueError('`features` needs to be of shape [bsz, n_views, ...]')
        if len(features.shape) == 2:
            features = features.unsqueeze(1)

        batch_size = features.shape[0]
        if labels is not None and mask is None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # Compute logits (cosine similarity / temperature)
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )

        # Numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # Mask out self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute log-prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class ColorInvariantCombinedLoss(nn.Module):
    """
    Hybrid Objective Function: ArcFace Angular Loss + Supervised Contrastive Loss.
    Encourages hyper-spherical class separation while enforcing invariant intra-class manifold clustering.
    """
    def __init__(self, in_features, num_classes, s=30.0, m=0.50, alpha=1.0, beta=0.5, temperature=0.07):
        super(ColorInvariantCombinedLoss, self).__init__()
        self.arc_loss = ArcFaceLoss(in_features, num_classes, s=s, m=m)
        self.supcon_loss = SupConLoss(temperature=temperature)
        self.alpha = alpha
        self.beta = beta

    def forward(self, embeddings, labels):
        arc_val, logits = self.arc_loss(embeddings, labels)
        supcon_val = self.supcon_loss(embeddings, labels)
        total_loss = self.alpha * arc_val + self.beta * supcon_val
        return total_loss, logits, {"arcface_loss": arc_val.item(), "supcon_loss": supcon_val.item()}
