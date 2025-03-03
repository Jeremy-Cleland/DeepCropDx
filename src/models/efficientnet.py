# EfficientNet-B0 model modifications
import torch
import torch.nn as nn
import torchvision.models as models


def create_efficientnet_model(num_classes, pretrained=True, freeze_backbone=True):
    """
    Create an EfficientNet model with a custom classifier

    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        freeze_backbone (bool): Whether to freeze the backbone layers

    Returns:
        torch.nn.Module: The EfficientNet model
    """
    # Load pretrained EfficientNet-B0
    model = models.efficientnet_b0(pretrained=pretrained)

    # Freeze backbone layers if specified
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=in_features, out_features=num_classes),
    )

    return model


def create_efficientnet_b3_model(num_classes, pretrained=True, freeze_backbone=True):
    """
    Create an EfficientNet-B3 model with a custom classifier
    """
    model = models.efficientnet_b3(pretrained=pretrained)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the classifier - avoid inplace operations
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=False),  # Changed from inplace=True
        nn.Linear(in_features=in_features, out_features=512),
        nn.ReLU(inplace=False),  # Changed from inplace=True
        nn.Dropout(p=0.2, inplace=False),  # Changed from inplace=True
        nn.Linear(in_features=512, out_features=num_classes),
    )

    return model
