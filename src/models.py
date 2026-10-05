"""
Neural Network Architectures for Color-Invariant Saree Design Recognition.
Combines Deep Feature Backbones with ArcFace Metric Learning & GeM Pooling.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM).
    Learns optimal pooling exponent p to extract spatial motif features better than standard GAP.
    """
    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-3))).pow(1.0 / self.p)


class ColorInvariantSareeNet(nn.Module):
    """
    Deep Architecture for Color-Invariant Textile Pattern Recognition.
    
    Supports:
    - ResNet34, ResNet50, MobileNetV3, ConvNeXt backbones.
    - GeM (Generalized Mean) pooling.
    - L2-normalized embedding bottleneck.
    - Optional Dual-Stream edge/color fusion.
    """
    def __init__(self, backbone_name="resnet34", embedding_size=512, pretrained=True):
        super(ColorInvariantSareeNet, self).__init__()
        self.backbone_name = backbone_name
        self.embedding_size = embedding_size

        if backbone_name == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            base = models.resnet34(weights=weights)
            in_features = base.fc.in_features
            self.backbone = nn.Sequential(*list(base.children())[:-2])
        elif backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base = models.resnet50(weights=weights)
            in_features = base.fc.in_features
            self.backbone = nn.Sequential(*list(base.children())[:-2])
        elif backbone_name == "mobilenet_v3":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            base = models.mobilenet_v3_small(weights=weights)
            in_features = base.classifier[0].in_features
            self.backbone = base.features
        elif backbone_name == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            base = models.convnext_tiny(weights=weights)
            in_features = base.classifier[2].in_features
            self.backbone = base.features
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.pooling = GeM(p=3.0)
        
        # Embedding Neck
        self.neck = nn.Sequential(
            nn.Linear(in_features, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size),
            nn.PReLU()
        )

    def extract_features(self, x):
        features = self.backbone(x)
        pooled = self.pooling(features).flatten(1)
        embeddings = self.neck(pooled)
        # L2 Normalization for Cosine Similarity metric space
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        return normalized_embeddings

    def forward(self, x):
        return self.extract_features(x)


class ArcMarginProduct(nn.Module):
    """
    ArcFace (Additive Angular Margin Loss) Classifier Head.
    Reference: Deng et al. "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input_embeddings, label=None):
        # Cosine theta = <X, W>
        cosine = F.linear(F.normalize(input_embeddings), F.normalize(self.weight))
        
        if label is None:
            return cosine * self.s

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).clamp(0, 1)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=input_embeddings.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return output
