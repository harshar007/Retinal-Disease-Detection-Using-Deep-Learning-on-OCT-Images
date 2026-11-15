"""
Module: src.models
Defines the Custom CNN built from scratch and Transfer Learning models:
- CustomOCTCNN (Conv2d, BatchNorm, ReLU, MaxPool, AdaptiveAvgPool, Dropout, Dense)
- ResNet18
- VGG16
- MobileNetV2
- DenseNet121
"""

import torch
import torch.nn as nn
from torchvision import models


class CustomOCTCNN(nn.Module):
    """
    Custom Convolutional Neural Network built from scratch using PyTorch.
    Features:
    - 4 Convolutional stages with Batch Normalization and ReLU
    - Max Pooling and Adaptive Average Pooling
    - Dropout regularization (0.4 and 0.3) to prevent overfitting
    - Multi-layer classification head for 4 OCT retinal disease categories
    """
    def __init__(self, num_classes=4, dropout_rate=0.4):
        super(CustomOCTCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Stage 1: 3 -> 32 channels, 224x224 -> 112x112
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Stage 2: 32 -> 64 channels, 112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Stage 3: 64 -> 128 channels, 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Stage 4: 128 -> 256 channels, 28x28 -> 14x14
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.75),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        feat = self.features(x)
        out = self.classifier(feat)
        return out


def build_resnet18(num_classes=4, pretrained=True, freeze_backbone=False):
    """ResNet18 transfer learning model with modified classification head."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


def build_vgg16(num_classes=4, pretrained=True, freeze_backbone=False):
    """VGG16 transfer learning model with modified classification head."""
    weights = models.VGG16_Weights.DEFAULT if pretrained else None
    model = models.vgg16(weights=weights)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


def build_mobilenet_v2(num_classes=4, pretrained=True, freeze_backbone=False):
    """MobileNetV2 lightweight transfer learning model for efficient edge inference."""
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


def build_densenet121(num_classes=4, pretrained=True, freeze_backbone=False):
    """DenseNet121 transfer learning model with dense feature reuse."""
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


SUPPORTED_MODELS = {
    "custom_cnn": CustomOCTCNN,
    "resnet18": build_resnet18,
    "mobilenet_v2": build_mobilenet_v2,
    "densenet121": build_densenet121,
    "vgg16": build_vgg16,
}


def build_model(model_name="resnet18", num_classes=4, pretrained=True, freeze_backbone=False):
    """
    Model factory function returning the requested architecture.
    """
    name = model_name.lower()
    if name == "custom_cnn":
        return CustomOCTCNN(num_classes=num_classes)
    elif name in SUPPORTED_MODELS:
        builder = SUPPORTED_MODELS[name]
        return builder(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Supported: {list(SUPPORTED_MODELS.keys())}")
