# ResNet50 model modifications
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models  # Add this import statement

# ResNet18 and ResNet50 weights
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
)


def create_resnet18_model(
    num_classes,
    use_weights=True,
    freeze_backbone=True,
    dropout_rate=0.3,
    add_hidden_layer=False,
    hidden_layer_size=512,
    **kwargs,
):
    weights = ResNet18_Weights.IMAGENET1K_V1 if use_weights else None
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features

    if add_hidden_layer:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=False),
            nn.Linear(in_features=in_features, out_features=hidden_layer_size),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout_rate / 2, inplace=False),
            nn.Linear(in_features=hidden_layer_size, out_features=num_classes),
        )
    else:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=False),
            nn.Linear(in_features=in_features, out_features=num_classes),
        )

    return model


def create_resnet50_model(
    num_classes,
    use_weights=True,
    freeze_backbone=True,
    dropout_rate=0.3,
    add_hidden_layer=True,  # ResNet50 usually has a hidden layer
    hidden_layer_size=512,
    **kwargs,
):
    """
    Create a ResNet-50 model with a custom classifier using the new 'weights' parameter.

    Args:
        num_classes (int): Number of output classes.
        use_weights (bool): If True, load pretrained weights (ResNet50_Weights.IMAGENET1K_V1).
        freeze_backbone (bool): Whether to freeze the backbone layers.
        dropout_rate (float): Dropout probability.
        add_hidden_layer (bool): Whether to add a hidden layer to the classifier.
        hidden_layer_size (int): Size of the hidden layer if used.

    Returns:
        torch.nn.Module: Modified ResNet-50 model.
    """
    weights = ResNet50_Weights.IMAGENET1K_V1 if use_weights else None
    model = models.resnet50(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features

    if add_hidden_layer:
        model.fc = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_layer_size),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout_rate, inplace=False),
            nn.Linear(in_features=hidden_layer_size, out_features=num_classes),
        )
    else:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=False),
            nn.Linear(in_features=in_features, out_features=num_classes),
        )

    return model


# Residual Attention Module for ResNet
class ResidualAttention(nn.Module):
    """
    Attention module for ResNet.
    """

    def __init__(self, channels, reduction=16):
        super(ResidualAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResNetWithAttention(nn.Module):
    """
    ResNet model with an attention mechanism.
    """

    def __init__(self, resnet_model, num_classes):
        super(ResNetWithAttention, self).__init__()
        # Use all layers except the final pooling and fc layer
        self.features = nn.Sequential(*list(resnet_model.children())[:-2])

        # Determine number of features from last conv layer
        # For ResNet50 and similar architectures, it's typically 2048.
        num_features = 2048

        self.attention = ResidualAttention(num_features)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def create_resnet_with_attention(
    num_classes,
    resnet_version=50,
    use_weights=True,
    freeze_backbone=True,
    dropout_rate=0.5,
    attention_reduction=16,
    **kwargs,
):
    """
    Create a ResNet model with an attention mechanism using the new 'weights' parameter.

    Args:
        num_classes (int): Number of output classes.
        resnet_version (int): ResNet version (18, 34, 50, 101, or 152).
        use_weights (bool): If True, load pretrained weights for the specified ResNet.
        freeze_backbone (bool): Whether to freeze the backbone layers.
        dropout_rate (float): Dropout probability for the classifier.
        attention_reduction (int): Reduction factor for the attention module.

    Returns:
        torch.nn.Module: Modified ResNet model with attention.
    """
    if resnet_version == 18:
        weights = ResNet18_Weights.IMAGENET1K_V1 if use_weights else None
        base_model = models.resnet18(weights=weights)
        num_features = 512  # ResNet18 uses 512 features in final layer
    elif resnet_version == 34:
        weights = ResNet34_Weights.IMAGENET1K_V1 if use_weights else None
        base_model = models.resnet34(weights=weights)
        num_features = 512  # ResNet34 uses 512 features in final layer
    elif resnet_version == 50:
        weights = ResNet50_Weights.IMAGENET1K_V1 if use_weights else None
        base_model = models.resnet50(weights=weights)
        num_features = 2048  # ResNet50 uses 2048 features in final layer
    elif resnet_version == 101:
        weights = ResNet101_Weights.IMAGENET1K_V1 if use_weights else None
        base_model = models.resnet101(weights=weights)
        num_features = 2048  # ResNet101 uses 2048 features in final layer
    elif resnet_version == 152:
        weights = ResNet152_Weights.IMAGENET1K_V1 if use_weights else None
        base_model = models.resnet152(weights=weights)
        num_features = 2048  # ResNet152 uses 2048 features in final layer
    else:
        raise ValueError(f"Unsupported ResNet version: {resnet_version}")

    if freeze_backbone:
        for param in base_model.parameters():
            param.requires_grad = False

    class ResNetWithAttention(nn.Module):
        """ResNet model with an attention mechanism."""

        def __init__(
            self, features, num_features, num_classes, dropout_rate, reduction
        ):
            super(ResNetWithAttention, self).__init__()
            # Use all layers except the final pooling and fc layer
            self.features = features

            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(num_features, num_features // reduction),
                nn.ReLU(inplace=False),
                nn.Linear(num_features // reduction, num_features),
                nn.Sigmoid(),
            )

            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate, inplace=False),
                nn.Linear(num_features, num_classes),
            )

        def forward(self, x):
            features = self.features(x)

            # Apply attention
            att = self.attention(features)
            att = att.view(att.size(0), att.size(1), 1, 1)
            features = features * att

            # Pooling and classification
            x = self.avg_pool(features)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    # Extract the feature extractor (everything except the final FC layer)
    features = nn.Sequential(*list(base_model.children())[:-2])

    # Create the attention model
    model = ResNetWithAttention(
        features=features,
        num_features=num_features,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        reduction=attention_reduction,
    )

    return model
