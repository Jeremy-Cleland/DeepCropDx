import torch
import torch.nn as nn
import torchvision.models as models

from torchvision.models import MobileNet_V2_Weights


def create_mobilenet_model(num_classes, use_weights=True, freeze_backbone=True):
    """
    Create a MobileNetV2 model with a custom classifier using the new 'weights' parameter.

    Args:
        num_classes (int): Number of output classes.
        use_weights (bool): If True, load pretrained weights (MobileNet_V2_Weights.IMAGENET1K_V1).
        freeze_backbone (bool): Whether to freeze the backbone layers.

    Returns:
        torch.nn.Module: Modified MobileNetV2 model.
    """
    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if use_weights else None
    model = models.mobilenet_v2(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the classifier (the second element in the classifier Sequential)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=in_features, out_features=num_classes),
    )
    return model


# MobileNet V3
from torchvision.models import MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights


def create_mobilenet_v3_model(
    num_classes, use_weights=True, freeze_backbone=True, model_size="small"
):
    """
    Create a MobileNetV3 model with a custom classifier using the new 'weights' parameter.

    Args:
        num_classes (int): Number of output classes.
        use_weights (bool): If True, load pretrained weights (MobileNet_V3_Small_Weights or MobileNet_V3_Large_Weights).
        freeze_backbone (bool): Whether to freeze the backbone layers.
        model_size (str): 'small' or 'large' model variant.

    Returns:
        torch.nn.Module: Modified MobileNetV3 model.
    """
    if model_size.lower() == "small":
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if use_weights else None
        model = models.mobilenet_v3_small(weights=weights)
    else:
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if use_weights else None
        model = models.mobilenet_v3_large(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # For MobileNetV3, classifier is structured differently:
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features=in_features, out_features=512),
        nn.Hardswish(inplace=True),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=512, out_features=num_classes),
    )
    return model


def create_mobilenet_model_with_attention(
    num_classes, use_weights=True, freeze_backbone=True
):
    """
    Create a MobileNetV2 model with a custom classifier and an attention mechanism.

    Args:
        num_classes (int): Number of output classes.
        use_weights (bool): If True, load pretrained weights (MobileNet_V2_Weights.IMAGENET1K_V1).
        freeze_backbone (bool): Whether to freeze the backbone layers (only freeze features).

    Returns:
        torch.nn.Module: Modified MobileNetV2 model with attention.
    """
    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if use_weights else None
    model = models.mobilenet_v2(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    feature_extractor = model.features

    class MobileNetWithAttention(nn.Module):
        def __init__(self, feature_extractor, num_classes):
            super(MobileNetWithAttention, self).__init__()
            self.features = feature_extractor
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            feature_dim = 1280  # Last channel size in MobileNetV2

            # Attention block
            self.attention = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim // 16, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature_dim // 16, feature_dim, kernel_size=1),
                nn.Sigmoid(),
            )

            self.classifier = nn.Sequential(
                nn.Dropout(p=0.2), nn.Linear(feature_dim, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            attention = self.attention(x)
            x = x * attention
            x = self.avg_pool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

    return MobileNetWithAttention(feature_extractor, num_classes)
