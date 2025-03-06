# EfficientNet-B0 model modifications
import torch
import torch.nn as nn
import torchvision.models as models


def create_efficientnet_model(
    num_classes,
    pretrained=True,
    freeze_backbone=True,
    dropout_rate=0.2,
    add_hidden_layer=False,
    hidden_layer_size=256,
    activation="relu",
    **kwargs
):
    """
    Create an EfficientNet-B0 model with a custom classifier.
    """
    from torchvision.models import EfficientNet_B0_Weights
    import torch.nn as nn

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    # Freeze backbone layers if specified
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Get input features size
    in_features = model.classifier[1].in_features  # typically 1280 for EfficientNet-B0

    # The crucial fix: EfficientNet in torchvision already has a pooling stage
    # We need to only replace the classifier part, not add another pooling sequence
    if add_hidden_layer:
        # Choose activation function
        act_fn = nn.ReLU(inplace=False)
        if activation == "leaky_relu":
            act_fn = nn.LeakyReLU(0.2, inplace=False)
        elif activation == "silu":
            act_fn = nn.SiLU(inplace=False)

        # Replace only the classifier part (keep the existing pooling)
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=False),
            nn.Linear(in_features=in_features, out_features=hidden_layer_size),
            act_fn,
            nn.Dropout(p=dropout_rate / 2, inplace=False),
            nn.Linear(in_features=hidden_layer_size, out_features=num_classes),
        )
    else:
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=False),
            nn.Linear(in_features=in_features, out_features=num_classes),
        )

    return model


def create_efficientnet_b3_model(
    num_classes,
    pretrained=True,
    freeze_backbone=True,
    dropout_rate=0.3,
    add_hidden_layer=False,
    hidden_layer_size=512,
    **kwargs
):
    from torchvision.models import EfficientNet_B3_Weights

    weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b3(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    else:
        # Enable gradient checkpointing to reduce memory usage
        model.set_grad_checkpointing(enable=True)

    in_features = model.classifier[1].in_features  # Typically 1536 for EfficientNet-B3

    # Remove the problematic nn.AdaptiveAvgPool2d and nn.Flatten layers
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
