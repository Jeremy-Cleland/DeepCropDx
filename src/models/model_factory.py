def create_model(
    model_type,
    num_classes,
    use_weights=True,
    freeze_backbone=True,
    dropout_rate=0.2,
    add_hidden_layer=False,
    hidden_layer_size=512,
    activation="relu",
    width_multiplier=1.0,
    attention_reduction=16,
    normalization="batch",
    **kwargs,
):
    """
    Factory method to create models based on model_type.

    Args:
        model_type (str): Type of model to create.
        num_classes (int): Number of output classes.
        use_weights (bool): Whether to load pretrained weights.
        freeze_backbone (bool): Whether to freeze backbone layers.
        dropout_rate (float): Dropout rate for classifier layers.
        **kwargs: Additional model-specific parameters.

    Returns:
        torch.nn.Module: Created model.
    """
    if model_type == "efficientnet":
        from src.models.efficientnet import create_efficientnet_model

        return create_efficientnet_model(
            num_classes=num_classes,
            pretrained=use_weights,
            freeze_backbone=freeze_backbone,
            **kwargs,  # Pass all other parameters as kwargs
        )

    elif model_type == "efficientnet_b3":
        from src.models.efficientnet import create_efficientnet_b3_model

        return create_efficientnet_b3_model(
            num_classes=num_classes,
            pretrained=use_weights,
            freeze_backbone=freeze_backbone,
            **kwargs,  # Pass all other parameters as kwargs
        )

    elif model_type == "mobilenet":
        from src.models.mobilenet import create_mobilenet_model

        return create_mobilenet_model(
            num_classes=num_classes,
            pretrained=use_weights,
            freeze_backbone=freeze_backbone,
            **kwargs,  # Pass all other parameters as kwargs
        )

    elif model_type == "mobilenet_v3_small":
        from src.models.mobilenet import create_mobilenet_v3_model

        return create_mobilenet_v3_model(
            num_classes=num_classes,
            pretrained=use_weights,
            freeze_backbone=freeze_backbone,
            **kwargs,  # Pass all other parameters as kwargs
        )

    elif model_type == "mobilenet_v3_large":
        from src.models.mobilenet import create_mobilenet_v3_model

        return create_mobilenet_v3_model(
            num_classes=num_classes,
            pretrained=use_weights,
            freeze_backbone=freeze_backbone,
            **kwargs,  # Pass all other parameters as kwargs
        )

    elif model_type == "mobilenet_attention":
        from src.models.mobilenet import create_mobilenet_model_with_attention

        return create_mobilenet_model_with_attention(
            num_classes=num_classes,
            pretrained=use_weights,
            freeze_backbone=freeze_backbone,
            **kwargs,  # Pass all other parameters as kwargs
        )

    elif model_type == "resnet18":
        from src.models.resnet import create_resnet18_model

        return create_resnet18_model(
            num_classes=num_classes,
            pretrained=use_weights,
            freeze_backbone=freeze_backbone,
            **kwargs,  # Pass all other parameters as kwargs
        )

    elif model_type == "resnet50":
        from src.models.resnet import create_resnet50_model

        return create_resnet50_model(
            num_classes=num_classes,
            pretrained=use_weights,
            freeze_backbone=freeze_backbone,
            **kwargs,  # Pass all other parameters as kwargs
        )

    elif model_type == "resnet_attention":
        from src.models.resnet import create_resnet_with_attention

        resnet_version = kwargs.get("resnet_version", 50)
        return create_resnet_with_attention(
            num_classes=num_classes,
            pretrained=use_weights,
            freeze_backbone=freeze_backbone,
            **kwargs,  # Pass all other parameters as kwargs
        )

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
