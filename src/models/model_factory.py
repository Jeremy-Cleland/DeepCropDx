# src/models/model_factory.py
def create_model(
    model_type, num_classes, pretrained=True, freeze_backbone=True, **kwargs
):
    """
    Factory method to create models based on model_type

    Args:
        model_type (str): Type of model to create
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        freeze_backbone (bool): Whether to freeze backbone layers
        **kwargs: Additional model-specific parameters

    Returns:
        model (nn.Module): Created model
    """
    if model_type == "efficientnet":
        from src.models.efficientnet import create_efficientnet_model

        return create_efficientnet_model(num_classes, pretrained, freeze_backbone)

    elif model_type == "efficientnet_b3":
        from src.models.efficientnet import create_efficientnet_b3_model

        return create_efficientnet_b3_model(num_classes, pretrained, freeze_backbone)

    elif model_type == "mobilenet":
        from src.models.mobilenet import create_mobilenet_model

        return create_mobilenet_model(num_classes, pretrained, freeze_backbone)

    elif model_type == "mobilenet_v3_small":
        from src.models.mobilenet import create_mobilenet_v3_model

        return create_mobilenet_v3_model(
            num_classes, pretrained, freeze_backbone, "small"
        )

    elif model_type == "mobilenet_v3_large":
        from src.models.mobilenet import create_mobilenet_v3_model

        return create_mobilenet_v3_model(
            num_classes, pretrained, freeze_backbone, "large"
        )

    elif model_type == "mobilenet_attention":
        from src.models.mobilenet import create_mobilenet_model_with_attention

        return create_mobilenet_model_with_attention(
            num_classes, pretrained, freeze_backbone
        )

    elif model_type == "resnet18":
        from src.models.resnet import create_resnet18_model

        return create_resnet18_model(num_classes, pretrained, freeze_backbone)

    elif model_type == "resnet50":
        from src.models.resnet import create_resnet50_model

        return create_resnet50_model(num_classes, pretrained, freeze_backbone)

    elif model_type == "resnet_attention":
        from src.models.resnet import create_resnet_with_attention

        resnet_version = kwargs.get("resnet_version", 50)
        return create_resnet_with_attention(
            num_classes, resnet_version, pretrained, freeze_backbone
        )

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
