# EfficientNet-B0 model modifications
import torch
import torch.nn as nn
import torchvision.models as models


def create_efficientnet_model(num_classes, pretrained=True, freeze_backbone=True):
    """
    Create an EfficientNet-B0 model with a custom classifier using the new API.

    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        freeze_backbone (bool): Whether to freeze the backbone layers

    Returns:
        torch.nn.Module: The EfficientNet-B0 model
    """
    from torchvision.models import EfficientNet_B0_Weights

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)

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
    Create an EfficientNet-B3 model with a custom classifier using the new API.

    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        freeze_backbone (bool): Whether to freeze the backbone layers

    Returns:
        torch.nn.Module: The EfficientNet-B3 model

    Notes:
        - With pretrained=True, uses IMAGENET1K_V1 weights (82.008% top-1, 96.054% top-5 accuracy)
        - The weights provide transforms with 320px resize, 300px center crop, and ImageNet normalization
    """
    from torchvision.models import EfficientNet_B3_Weights

    # Use explicit weights enum instead of boolean pretrained flag
    weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b3(weights=weights)

    # Freeze backbone layers if specified
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the classifier - avoid inplace operations where specified
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=False),
        nn.Linear(in_features=in_features, out_features=512),
        nn.ReLU(inplace=False),
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(in_features=512, out_features=num_classes),
    )

    return model
