# MobileNetV3 model modifications
import torch
import torch.nn as nn
import torchvision.models as models


def create_mobilenet_model(num_classes, pretrained=True, freeze_backbone=True):
    """
    Create a MobileNetV2 model with a custom classifier

    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        freeze_backbone (bool): Whether to freeze the backbone layers

    Returns:
        torch.nn.Module: The MobileNetV2 model
    """
    # Load pretrained MobileNetV2
    model = models.mobilenet_v2(pretrained=pretrained)

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


def create_mobilenet_v3_model(
    num_classes, pretrained=True, freeze_backbone=True, model_size="small"
):
    """
    Create a MobileNetV3 model with a custom classifier

    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        freeze_backbone (bool): Whether to freeze the backbone layers
        model_size (str): Size of the model, either 'small' or 'large'

    Returns:
        torch.nn.Module: The MobileNetV3 model
    """
    # Load pretrained MobileNetV3
    if model_size.lower() == "small":
        model = models.mobilenet_v3_small(pretrained=pretrained)
    else:
        model = models.mobilenet_v3_large(pretrained=pretrained)

    # Freeze backbone layers if specified
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the classifier
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features=in_features, out_features=512),
        nn.Hardswish(inplace=True),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=512, out_features=num_classes),
    )

    return model


def create_mobilenet_model_with_attention(
    num_classes, pretrained=True, freeze_backbone=True
):
    """
    Create a MobileNetV2 model with a custom classifier and attention mechanism

    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        freeze_backbone (bool): Whether to freeze the backbone layers

    Returns:
        torch.nn.Module: The MobileNetV2 model with attention
    """
    # Load pretrained MobileNetV2
    model = models.mobilenet_v2(pretrained=pretrained)

    # Freeze backbone layers if specified
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # Extract the feature extractor
    feature_extractor = model.features

    # Create a new model with attention
    class MobileNetWithAttention(nn.Module):
        def __init__(self, feature_extractor, num_classes):
            super(MobileNetWithAttention, self).__init__()
            self.features = feature_extractor
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

            # Attention block
            feature_dim = 1280  # Last channel size in MobileNetV2
            self.attention = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim // 16, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature_dim // 16, feature_dim, kernel_size=1),
                nn.Sigmoid(),
            )

            # Classifier
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.2), nn.Linear(feature_dim, num_classes)
            )

        def forward(self, x):
            x = self.features(x)

            # Apply attention
            attention = self.attention(x)
            x = x * attention

            # Global average pooling and classify
            x = self.avg_pool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)

            return x

    return MobileNetWithAttention(feature_extractor, num_classes)
