# ResNet50 model modifications
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


def create_resnet18_model(num_classes, pretrained=True, freeze_backbone=True):
    """
    Create a ResNet-18 model with a custom classifier

    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        freeze_backbone (bool): Whether to freeze the backbone layers

    Returns:
        torch.nn.Module: The ResNet-18 model
    """
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=pretrained)

    # Freeze backbone layers if specified
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the classifier (fc layer)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3), nn.Linear(in_features=in_features, out_features=num_classes)
    )

    return model


def create_resnet50_model(num_classes, pretrained=True, freeze_backbone=True):
    """
    Create a ResNet-50 model with a custom classifier

    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        freeze_backbone (bool): Whether to freeze the backbone layers

    Returns:
        torch.nn.Module: The ResNet-50 model
    """
    # Load pretrained ResNet-50
    model = models.resnet50(pretrained=pretrained)

    # Freeze backbone layers if specified
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the classifier (fc layer)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features=in_features, out_features=512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(in_features=512, out_features=num_classes),
    )

    return model


class ResidualAttention(nn.Module):
    """
    Attention module for ResNet
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
    ResNet model with attention mechanism
    """

    def __init__(self, resnet_model, num_classes):
        super(ResNetWithAttention, self).__init__()
        # Remove the original fully connected layer
        self.features = nn.Sequential(*list(resnet_model.children())[:-2])

        # Get the number of features from the last convolutional layer
        if isinstance(resnet_model, models.ResNet):
            if resnet_model.layer4[0].downsample is not None:
                num_features = resnet_model.layer4[0].downsample[0].out_channels
            else:
                num_features = resnet_model.layer4[0].conv1.out_channels * 4
        else:
            # Default for ResNet50 and similar
            num_features = 2048

        # Add attention module
        self.attention = ResidualAttention(num_features)

        # Global average pooling and classifier
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
    num_classes, resnet_version=50, pretrained=True, freeze_backbone=True
):
    """
    Create a ResNet model with attention mechanism

    Args:
        num_classes (int): Number of output classes
        resnet_version (int): ResNet version (18, 34, 50, 101, or 152)
        pretrained (bool): Whether to use pretrained weights
        freeze_backbone (bool): Whether to freeze the backbone layers

    Returns:
        torch.nn.Module: The ResNet model with attention
    """
    # Load appropriate ResNet model
    if resnet_version == 18:
        base_model = models.resnet18(pretrained=pretrained)
    elif resnet_version == 34:
        base_model = models.resnet34(pretrained=pretrained)
    elif resnet_version == 50:
        base_model = models.resnet50(pretrained=pretrained)
    elif resnet_version == 101:
        base_model = models.resnet101(pretrained=pretrained)
    elif resnet_version == 152:
        base_model = models.resnet152(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported ResNet version: {resnet_version}")

    # Freeze backbone if specified
    if freeze_backbone:
        for param in base_model.parameters():
            param.requires_grad = False

    # Create the model with attention
    model = ResNetWithAttention(base_model, num_classes)

    return model
