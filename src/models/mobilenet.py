import torch
import torch.nn as nn
import torchvision.models as models

from torchvision.models import MobileNet_V2_Weights


def create_mobilenet_model(
    num_classes,
    use_weights=True,
    freeze_backbone=True,
    dropout_rate=0.2,
    add_hidden_layer=False,
    hidden_layer_size=512,
    **kwargs
):
    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if use_weights else None
    model = models.mobilenet_v2(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Get input features size
    in_features = model.classifier[1].in_features

    # Replace the classifier with optional hidden layer
    if add_hidden_layer:
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=False),
            nn.Linear(in_features=in_features, out_features=hidden_layer_size),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout_rate / 2, inplace=False),
            nn.Linear(in_features=hidden_layer_size, out_features=num_classes),
        )
    else:
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=False),
            nn.Linear(in_features=in_features, out_features=num_classes),
        )

    return model


# MobileNet V3
from torchvision.models import MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights


def create_mobilenet_v3_model(
    num_classes,
    use_weights=True,
    freeze_backbone=True,
    model_size="small",
    dropout_rate=0.2,
    add_hidden_layer=True,
    hidden_layer_size=512,
    **kwargs
):
    if model_size.lower() == "small":
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if use_weights else None
        model = models.mobilenet_v3_small(weights=weights)
    else:
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if use_weights else None
        model = models.mobilenet_v3_large(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Get input features size
    in_features = model.classifier[0].in_features

    # Create classifier - MobileNetV3 always has a hidden layer in typical use
    model.classifier = nn.Sequential(
        nn.Linear(in_features=in_features, out_features=hidden_layer_size),
        nn.Hardswish(inplace=False),
        nn.Dropout(p=dropout_rate, inplace=False),
        nn.Linear(in_features=hidden_layer_size, out_features=num_classes),
    )

    return model


def create_mobilenet_model_with_attention(
    num_classes,
    use_weights=True,
    freeze_backbone=True,
    dropout_rate=0.2,
    attention_reduction=16,
    **kwargs
):
    """
    Create a MobileNetV2 model with a custom classifier and an attention mechanism.

    Args:
        num_classes (int): Number of output classes.
        use_weights (bool): If True, load pretrained weights (MobileNet_V2_Weights.IMAGENET1K_V1).
        freeze_backbone (bool): Whether to freeze the backbone layers (only freeze features).
        dropout_rate (float): Dropout probability.
        attention_reduction (int): Reduction factor for the attention module.

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
        def __init__(self, feature_extractor, num_classes, dropout_rate, reduction):
            super(MobileNetWithAttention, self).__init__()
            self.features = feature_extractor
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            feature_dim = 1280  # Last channel size in MobileNetV2

            # Attention block
            self.attention = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim // reduction, kernel_size=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(feature_dim // reduction, feature_dim, kernel_size=1),
                nn.Sigmoid(),
            )

            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate, inplace=False),
                nn.Linear(feature_dim, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            attention = self.attention(x)
            x = x * attention
            x = self.avg_pool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

    return MobileNetWithAttention(
        feature_extractor, num_classes, dropout_rate, attention_reduction
    )
