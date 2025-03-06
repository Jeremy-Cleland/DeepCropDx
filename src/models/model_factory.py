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
    test_forward=True,
    **kwargs,
):
    """
    Create a model of the specified type with proper configuration.

    Args:
        model_type (str): Type of model to create
        num_classes (int): Number of output classes
        use_weights (bool): Whether to use pretrained weights
        freeze_backbone (bool): Whether to freeze backbone layers
        dropout_rate (float): Dropout rate for classification head
        add_hidden_layer (bool): Whether to add a hidden layer in classifier
        hidden_layer_size (int): Size of hidden layer if used
        activation (str): Activation function to use
        width_multiplier (float): Width multiplier for mobilenet models
        attention_reduction (int): Reduction factor for attention models
        normalization (str): Normalization type (batch, instance, layer, none)
        test_forward (bool): Whether to test forward pass with dummy input
        **kwargs: Additional parameters to pass to the model creation function

    Returns:
        torch.nn.Module: Created model
    """
    import torch

    model = None

    try:
        if model_type == "efficientnet":
            from src.models.efficientnet import create_efficientnet_model

            model = create_efficientnet_model(
                num_classes=num_classes,
                pretrained=use_weights,
                freeze_backbone=freeze_backbone,
                dropout_rate=dropout_rate,
                add_hidden_layer=add_hidden_layer,
                hidden_layer_size=hidden_layer_size,
                activation=activation,
                normalization=normalization,
                **kwargs,
            )

        elif model_type == "efficientnet_b3":
            from src.models.efficientnet import create_efficientnet_b3_model

            model = create_efficientnet_b3_model(
                num_classes=num_classes,
                pretrained=use_weights,
                freeze_backbone=freeze_backbone,
                dropout_rate=dropout_rate,
                add_hidden_layer=add_hidden_layer,
                hidden_layer_size=hidden_layer_size,
                activation=activation,
                normalization=normalization,
                **kwargs,
            )

        elif model_type == "mobilenet":
            from src.models.mobilenet import create_mobilenet_model

            model = create_mobilenet_model(
                num_classes=num_classes,
                use_weights=use_weights,
                freeze_backbone=freeze_backbone,
                dropout_rate=dropout_rate,
                add_hidden_layer=add_hidden_layer,
                hidden_layer_size=hidden_layer_size,
                width_multiplier=width_multiplier,
                **kwargs,
            )

        elif model_type == "mobilenet_v3_small":
            from src.models.mobilenet import create_mobilenet_v3_model

            model = create_mobilenet_v3_model(
                num_classes=num_classes,
                use_weights=use_weights,
                freeze_backbone=freeze_backbone,
                dropout_rate=dropout_rate,
                add_hidden_layer=add_hidden_layer,
                hidden_layer_size=hidden_layer_size,
                model_size="small",
                **kwargs,
            )

        elif model_type == "mobilenet_v3_large":
            from src.models.mobilenet import create_mobilenet_v3_model

            model = create_mobilenet_v3_model(
                num_classes=num_classes,
                use_weights=use_weights,
                freeze_backbone=freeze_backbone,
                dropout_rate=dropout_rate,
                add_hidden_layer=add_hidden_layer,
                hidden_layer_size=hidden_layer_size,
                model_size="large",
                **kwargs,
            )

        elif model_type == "mobilenet_attention":
            from src.models.mobilenet import create_mobilenet_model_with_attention

            model = create_mobilenet_model_with_attention(
                num_classes=num_classes,
                use_weights=use_weights,
                freeze_backbone=freeze_backbone,
                dropout_rate=dropout_rate,
                attention_reduction=attention_reduction,
                **kwargs,
            )

        elif model_type == "resnet18":
            from src.models.resnet import create_resnet18_model

            model = create_resnet18_model(
                num_classes=num_classes,
                use_weights=use_weights,
                freeze_backbone=freeze_backbone,
                dropout_rate=dropout_rate,
                add_hidden_layer=add_hidden_layer,
                hidden_layer_size=hidden_layer_size,
                **kwargs,
            )

        elif model_type == "resnet50":
            from src.models.resnet import create_resnet50_model

            model = create_resnet50_model(
                num_classes=num_classes,
                use_weights=use_weights,
                freeze_backbone=freeze_backbone,
                dropout_rate=dropout_rate,
                add_hidden_layer=add_hidden_layer,
                hidden_layer_size=hidden_layer_size,
                **kwargs,
            )

        elif model_type == "resnet_attention":
            from src.models.resnet import create_resnet_with_attention

            resnet_version = kwargs.get("resnet_version", 50)
            model = create_resnet_with_attention(
                num_classes=num_classes,
                resnet_version=resnet_version,
                use_weights=use_weights,
                freeze_backbone=freeze_backbone,
                dropout_rate=dropout_rate,
                attention_reduction=attention_reduction,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Add model type attribute for easier identification later
        model.model_type = model_type

        # Test forward pass with dummy input
        if test_forward:
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                output = model(dummy_input)
                expected_shape = (1, num_classes)
                if output.shape != expected_shape:
                    raise RuntimeError(
                        f"Model output shape mismatch: got {output.shape}, expected {expected_shape}"
                    )

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model created: {model_type}")
        print(f"Total parameters: {total_params:,}")
        print(
            f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.1%})"
        )

        return model

    except Exception as e:
        error_msg = f"Error creating {model_type} model: {str(e)}"
        print(f"ERROR: {error_msg}")
        import traceback

        traceback.print_exc()
        raise RuntimeError(error_msg)
