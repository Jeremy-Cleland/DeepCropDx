# Visualization functions (e.g., Grad-CAM, plotting)
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.cm as cm
import pandas as pd


def visualize_batch(images, labels, predictions=None, class_names=None, max_images=16):
    """
    Visualize a batch of images with their labels and predictions

    Args:
        images (torch.Tensor): Batch of images (B, C, H, W)
        labels (torch.Tensor): True labels
        predictions (torch.Tensor, optional): Predicted labels
        class_names (list, optional): List of class names
        max_images (int): Maximum number of images to display
    """
    # Convert from tensor to numpy array
    if isinstance(images, torch.Tensor):
        # Move to CPU if needed
        images = images.cpu().numpy()
        labels = labels.cpu().numpy()
        if predictions is not None:
            predictions = predictions.cpu().numpy()

    # Limit the number of images
    n_images = min(len(images), max_images)
    images = images[:n_images]
    labels = labels[:n_images]

    if predictions is not None:
        predictions = predictions[:n_images]

    # Compute grid dimensions
    n_cols = min(4, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols

    plt.figure(figsize=(15, 3 * n_rows))

    # Denormalize images
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i in range(n_images):
        plt.subplot(n_rows, n_cols, i + 1)

        # Denormalize and convert to appropriate format for display
        img = images[i].transpose(1, 2, 0) * std + mean
        img = np.clip(img, 0, 1)

        plt.imshow(img)

        if class_names:
            label_name = class_names[labels[i]]
            title = f"True: {label_name}"

            if predictions is not None:
                pred_name = class_names[predictions[i]]
                title += f"\nPred: {pred_name}"

                # Add color to indicate correct/incorrect prediction
                if predictions[i] == labels[i]:
                    plt.gca().spines["bottom"].set_color("green")
                    plt.gca().spines["top"].set_color("green")
                    plt.gca().spines["right"].set_color("green")
                    plt.gca().spines["left"].set_color("green")
                    plt.gca().spines["bottom"].set_linewidth(5)
                    plt.gca().spines["top"].set_linewidth(5)
                    plt.gca().spines["right"].set_linewidth(5)
                    plt.gca().spines["left"].set_linewidth(5)
                else:
                    plt.gca().spines["bottom"].set_color("red")
                    plt.gca().spines["top"].set_color("red")
                    plt.gca().spines["right"].set_color("red")
                    plt.gca().spines["left"].set_color("red")
                    plt.gca().spines["bottom"].set_linewidth(5)
                    plt.gca().spines["top"].set_linewidth(5)
                    plt.gca().spines["right"].set_linewidth(5)
                    plt.gca().spines["left"].set_linewidth(5)
        else:
            title = f"Label: {labels[i]}"
            if predictions is not None:
                title += f"\nPred: {predictions[i]}"

        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    return plt.gcf()


def plot_confusion_matrix(
    y_true, y_pred, class_names, normalize=True, figsize=(12, 10)
):
    """
    Plot confusion matrix

    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        class_names (list): List of class names
        normalize (bool): Whether to normalize the confusion matrix
        figsize (tuple): Figure size

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    return plt.gcf()


def plot_classification_report(y_true, y_pred, class_names):
    """
    Plot classification report as a heatmap

    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        class_names (list): List of class names

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Get classification report as a dictionary
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )

    # Convert to DataFrame
    df = pd.DataFrame(report).T

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    heatmap = sns.heatmap(df.iloc[:-3, :-1], annot=True, cmap="Blues", fmt=".2f")
    plt.title("Classification Report")
    plt.tight_layout()

    return plt.gcf()


def plot_training_history(history):
    """
    Plot training history

    Args:
        history (dict): Dictionary containing training history

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot training and validation loss
    ax1.plot(history.get("train_loss", []), label="Training Loss")
    ax1.plot(history.get("val_loss", []), label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True)

    # Plot training and validation metrics
    metrics = []
    for key in history.keys():
        if key not in ["train_loss", "val_loss"] and "train_" in key:
            metric_name = key.replace("train_", "")
            if f"val_{metric_name}" in history:
                metrics.append(metric_name)

    for metric in metrics:
        ax2.plot(history.get(f"train_{metric}", []), label=f"Training {metric}")
        ax2.plot(history.get(f"val_{metric}", []), label=f"Validation {metric}")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Metric Value")
    ax2.set_title("Training and Validation Metrics")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    return fig


# Replace the GradCAM class implementation (around line 200-300) with this improved version:


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.handles = []
        self.gradients = None
        self.activations = None
        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # Register hooks
        self.handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.handles.append(
            self.target_layer.register_full_backward_hook(backward_hook)
        )

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def generate_cam(self, input_tensor, target_class=None):
        # Important: Set model to eval mode but enable gradients
        was_training = self.model.training
        self.model.eval()

        # Reset gradients and activations
        self.gradients = None
        self.activations = None

        # Enable gradients even in eval mode
        with torch.set_grad_enabled(True):
            # Forward pass
            output = self.model(input_tensor)

            # Clear existing gradients
            self.model.zero_grad()

            # If no target class, use predicted class
            if target_class is None:
                target_class = output.argmax(dim=1).item()

            # For numerical stability - get the target score
            target_score = output[0, target_class]

            # Backward pass to get gradients
            target_score.backward(retain_graph=True)

            # Check if gradients were captured
            if self.gradients is None:
                print(
                    "Warning: No gradients captured! Target layer may not be suitable."
                )
                # Return blank cam
                return torch.zeros_like(input_tensor[0, 0]).detach().cpu().numpy()

            # Create heatmap from gradients and activations
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * self.activations).sum(dim=1, keepdim=True)

            # Apply ReLU (only positive contributions)
            cam = torch.relu(cam)

            # Resize to input size
            cam = torch.nn.functional.interpolate(
                cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False
            )[0, 0]

            # Normalize the CAM
            if cam.max() > cam.min():
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

            # Restore model's training state
            self.model.train(was_training)

            return cam.detach().cpu().numpy()

    def __del__(self):
        self.remove_hooks()


def get_target_layer_for_model(model, model_name):
    """
    Returns the target layer for GradCAM visualization based on the model architecture.

    Args:
        model: The PyTorch model
        model_name: Name of the model architecture (e.g., 'efficientnet', 'resnet50')

    Returns:
        The target layer for GradCAM visualization
    """
    print(f"Finding GradCAM target layer for model type: {model_name}")
    target_layer = None

    # Handle EfficientNet models
    if "efficientnet" in model_name.lower():
        # ...existing EfficientNet handling code...
        if hasattr(model, "features"):
            # Calculate the layer index based on the variant
            if "b3" in model_name.lower():
                # For B3, we use an earlier block (around 75% through the network)
                layer_idx = int(len(model.features._modules) * 0.75)
            elif "b0" in model_name.lower() or "efficientnet" == model_name.lower():
                # For B0 or generic EfficientNet, use the second-to-last block
                layer_idx = len(model.features._modules) - 2
            else:
                # For other variants, use a block about 80% through the network
                layer_idx = int(len(model.features._modules) * 0.8)

            print(
                f"Selected layer index {layer_idx} for {model_name} (out of {len(model.features._modules)})"
            )
            target_module = model.features._modules[str(layer_idx)]

            # Find a suitable convolutional layer
            for name, module in target_module.named_modules():
                if isinstance(module, torch.nn.Conv2d) and module.kernel_size[0] > 1:
                    target_layer = module
                    break

            # Fallbacks for EfficientNet if no suitable layer found
            if (
                target_layer is None
                and hasattr(target_module, "project")
                and hasattr(target_module.project, "0")
            ):
                target_layer = target_module.project[0]

            if target_layer is None:
                for name, module in target_module.named_modules():
                    if isinstance(module, torch.nn.Conv2d):
                        target_layer = module
                        break

    # Handle ResNet models (both standard and with attention)
    elif "resnet" in model_name.lower():
        if "attention" in model_name.lower():
            # Handle ResNet with attention mechanism
            if hasattr(model, "features") and hasattr(model.features, "layer4"):
                target_layer = model.features.layer4[-1]
            elif hasattr(model, "attention") and hasattr(model, "features"):
                # Find the last convolutional layer before attention
                for name, module in reversed(list(model.features.named_modules())):
                    if isinstance(module, torch.nn.Conv2d):
                        target_layer = module
                        break
        else:
            # Standard ResNet
            if hasattr(model, "layer4"):
                target_layer = model.layer4[-1]
                print(f"Using standard ResNet layer4[-1] for {model_name}")

    # Handle MobileNet variants
    elif "mobilenet" in model_name.lower():
        if "v3" in model_name.lower():
            # MobileNetV3
            if hasattr(model, "features"):
                # Find the last conv layer
                for i in range(len(model.features) - 1, -1, -1):
                    if isinstance(model.features[i], torch.nn.Conv2d):
                        target_layer = model.features[i]
                        print(f"Found MobileNetV3 conv layer at index {i}")
                        break

                # If no conv found, try getting the last block's conv
                if target_layer is None and len(model.features) > 0:
                    last_block = model.features[-1]
                    for name, module in last_block.named_modules():
                        if (
                            isinstance(module, torch.nn.Conv2d)
                            and module.kernel_size[0] > 1
                        ):
                            target_layer = module
                            print(f"Using MobileNetV3 last block conv: {name}")
                            break

        elif "attention" in model_name.lower():
            # MobileNet with attention
            if hasattr(model, "features"):
                # Find the conv layer right before attention mechanism
                for i in range(len(model.features) - 1, -1, -1):
                    if isinstance(model.features[i], torch.nn.Conv2d):
                        target_layer = model.features[i]
                        print(f"Found MobileNet with attention conv layer at index {i}")
                        break

                if target_layer is None and hasattr(model, "attention"):
                    # Try to find a conv in the main branch before attention
                    for name, module in model.named_modules():
                        if (
                            "attention" not in name
                            and isinstance(module, torch.nn.Conv2d)
                            and module.kernel_size[0] > 1
                        ):
                            target_layer = module
                            print(f"Using MobileNet conv before attention: {name}")
                            break

        else:
            # Standard MobileNetV2
            if hasattr(model, "features"):
                # Get the last feature block with a large enough kernel
                for i in range(len(model.features) - 1, -1, -1):
                    if (
                        isinstance(model.features[i], torch.nn.Conv2d)
                        and model.features[i].kernel_size[0] > 1
                    ):
                        target_layer = model.features[i]
                        print(f"Found MobileNetV2 conv layer at index {i}")
                        break

                # Fallback to any conv in features
                if target_layer is None:
                    for i in range(len(model.features) - 1, -1, -1):
                        if isinstance(model.features[i], torch.nn.Conv2d):
                            target_layer = model.features[i]
                            print(f"Fallback: using MobileNetV2 conv at index {i}")
                            break

                # Last resort: use the last module in features
                if target_layer is None:
                    # Try to find a suitable conv within the last module
                    last_module = model.features[-1]
                    for name, module in last_module.named_modules():
                        if isinstance(module, torch.nn.Conv2d):
                            target_layer = module
                            print(f"Using conv from last module: {name}")
                            break

    # If we still haven't found a suitable layer, use generic approach
    if target_layer is None:
        print("Using fallback approach to find a suitable layer")

        # Try to find any conv layer with kernel size > 1
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) and module.kernel_size[0] > 1:
                target_layer = module
                print(f"Found conv layer with kernel size > 1: {name}")
                break

        # Fallback to any conv layer
        if target_layer is None:
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module
                    print(f"Last resort: using any conv layer: {name}")
                    break

    if target_layer is None:
        raise ValueError(
            f"Could not find suitable layer for GradCAM in model: {model_name}"
        )

    print(f"Selected target layer: {target_layer}")
    return target_layer


def visualize(
    self, img_path, target_class=None, alpha=0.5, use_rgb=True, save_path=None
):
    """
    Visualize Grad-CAM on an image

    Args:
        img_path (str): Path to input image
        target_class (int, optional): Target class index
        alpha (float): Transparency factor for heatmap overlay
        use_rgb (bool): Whether to use RGB colormap
        save_path (str, optional): Path to save the visualization

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Load and preprocess image
    img = Image.open(img_path).convert("RGB")
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(img).unsqueeze(0)

    # Generate CAM
    cam = self.generate_cam(input_tensor, target_class)

    # Resize CAM to match input image size
    img_np = np.array(img.resize((224, 224)))
    cam_resized = np.uint8(255 * cam)

    # Apply colormap to CAM
    if use_rgb:
        heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    else:
        heatmap = plt.cm.jet(cam_resized)[:, :, :3] * 255

    # Overlay heatmap on image
    superimposed_img = heatmap * alpha + img_np * (1 - alpha)
    superimposed_img = np.uint8(superimposed_img)

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.subplot(131)
    plt.imshow(img_np)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(132)
    plt.imshow(cam, cmap="jet")
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")

    plt.subplot(133)
    plt.imshow(superimposed_img)
    plt.title("Grad-CAM Overlay")
    plt.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    return plt.gcf()


def visualize_model_predictions(model, dataloader, class_names, device, num_images=12):
    """
    Visualize model predictions on a batch of images

    Args:
        model (torch.nn.Module): Trained model
        dataloader (torch.utils.data.DataLoader): DataLoader
        class_names (list): List of class names
        device (torch.device): Device to run the model on
        num_images (int): Number of images to visualize

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(15, 10))

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 3 + 1, 3, images_so_far)
                ax.axis("off")

                # Denormalize images
                img = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)

                ax.imshow(img)

                # Add green/red border for correct/incorrect predictions
                if preds[j] == labels[j]:
                    color = "green"
                else:
                    color = "red"

                ax.spines["bottom"].set_color(color)
                ax.spines["top"].set_color(color)
                ax.spines["right"].set_color(color)
                ax.spines["left"].set_color(color)
                ax.spines["bottom"].set_linewidth(5)
                ax.spines["top"].set_linewidth(5)
                ax.spines["right"].set_linewidth(5)
                ax.spines["left"].set_linewidth(5)

                ax.set_title(
                    f"Pred: {class_names[preds[j]]}\nTrue: {class_names[labels[j]]}"
                )

                if images_so_far == num_images:
                    return fig

    return fig


def plot_class_distribution(dataset, class_names):
    """
    Plot class distribution in a dataset

    Args:
        dataset (torch.utils.data.Dataset): Dataset
        class_names (list): List of class names

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Count instances per class
    class_counts = {}
    for _, label in dataset:
        label_name = class_names[label]
        class_counts[label_name] = class_counts.get(label_name, 0) + 1

    # Sort by number of instances
    sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    classes, counts = zip(*sorted_counts)

    # Create plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, counts, color="skyblue")

    # Rotate x-axis labels if there are many classes
    if len(classes) > 5:
        plt.xticks(rotation=45, ha="right")

    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Instances")

    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{height}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    return plt.gcf()


def plot_model_feature_maps(
    model, image_tensor, layer_names, num_filters=8, figsize=(15, 10)
):
    """
    Plot feature maps from intermediate layers of a model

    Args:
        model (torch.nn.Module): Trained model
        image_tensor (torch.Tensor): Input image tensor (1, C, H, W)
        layer_names (list): List of layer names to visualize
        num_filters (int): Number of filters to display per layer
        figsize (tuple): Figure size

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Set up hooks to get intermediate activations
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()

        return hook

    # Register hooks
    handles = []
    for name, module in model.named_modules():
        if name in layer_names:
            handles.append(module.register_forward_hook(get_activation(name)))

    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(image_tensor)

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Plot feature maps
    plt.figure(figsize=figsize)

    for i, layer_name in enumerate(layer_names):
        if layer_name in activations:
            activation = activations[layer_name][0].cpu().numpy()
            num_channels = min(num_filters, activation.shape[0])

            for j in range(num_channels):
                plt.subplot(len(layer_names), num_filters, i * num_filters + j + 1)
                plt.imshow(activation[j], cmap="viridis")
                plt.title(f"{layer_name}\nFilter {j}")
                plt.axis("off")

    plt.tight_layout()
    return plt.gcf()


def visualize_activation_maximization(
    model, class_idx, img_size=224, device="cpu", iterations=100, lr=0.1
):
    """
    Visualize activation maximization for a specific class

    Args:
        model (torch.nn.Module): Trained model
        class_idx (int): Class index to visualize
        img_size (int): Image size
        device (str): Device to run the optimization on
        iterations (int): Number of optimization iterations
        lr (float): Learning rate

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Initialize a random image
    img = torch.randn(1, 3, img_size, img_size, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([img], lr=lr)

    # Optimize the image to maximize the class score
    model.eval()
    for i in range(iterations):
        optimizer.zero_grad()
        output = model(img)
        loss = -output[0, class_idx]
        loss.backward()
        optimizer.step()

        # Clip image values to valid range
        img.data = torch.clamp(img.data, 0, 1)

    # Convert optimized image for visualization
    img_viz = img.cpu().detach().numpy()[0].transpose(1, 2, 0)
    img_viz = np.clip(img_viz, 0, 1)

    # Create plot
    plt.figure(figsize=(8, 8))
    plt.imshow(img_viz)
    plt.title(f"Activation Maximization for Class {class_idx}")
    plt.axis("off")

    return plt.gcf()


def plot_examples_of_misclassifications(
    model, dataloader, class_names, device, num_examples=12
):
    """
    Plot examples of misclassified images

    Args:
        model (torch.nn.Module): Trained model
        dataloader (torch.utils.data.DataLoader): DataLoader
        class_names (list): List of class names
        device (torch.device): Device to run the model on
        num_examples (int): Number of misclassified examples to show

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Find misclassified examples
            misclassified_mask = preds != labels
            if torch.any(misclassified_mask):
                misclassified_idx = torch.where(misclassified_mask)[0]

                for idx in misclassified_idx:
                    misclassified_images.append(inputs[idx].cpu())
                    misclassified_labels.append(labels[idx].cpu())
                    misclassified_preds.append(preds[idx].cpu())

                    if len(misclassified_images) >= num_examples:
                        break

            if len(misclassified_images) >= num_examples:
                break

    # Convert to tensors
    misclassified_images = torch.stack(misclassified_images[:num_examples])
    misclassified_labels = torch.stack(misclassified_labels[:num_examples])
    misclassified_preds = torch.stack(misclassified_preds[:num_examples])

    # Visualize misclassified examples
    return visualize_batch(
        misclassified_images,
        misclassified_labels,
        misclassified_preds,
        class_names,
        max_images=num_examples,
    )


def visualize_tsne(
    features, labels, class_names=None, perplexity=30, n_iter=1000, figsize=(10, 8)
):
    """
    Visualize t-SNE projection of features

    Args:
        features (np.ndarray): Feature vectors (N, D)
        labels (np.ndarray): Labels (N,)
        class_names (list, optional): List of class names
        perplexity (int): Perplexity parameter for t-SNE
        n_iter (int): Number of iterations for t-SNE
        figsize (tuple): Figure size

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    from sklearn.manifold import TSNE

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    features_tsne = tsne.fit_transform(features)

    # Plot
    plt.figure(figsize=figsize)

    # Create scatter plot with different colors for each class
    unique_labels = np.unique(labels)

    for label in unique_labels:
        idx = labels == label

        if class_names is not None:
            label_name = class_names[label]
        else:
            label_name = f"Class {label}"

        plt.scatter(
            features_tsne[idx, 0], features_tsne[idx, 1], label=label_name, alpha=0.7
        )

    plt.legend()
    plt.title("t-SNE Visualization of Features")
    plt.tight_layout()

    return plt.gcf()


def save_all_visualizations(model, dataloaders, class_names, device, output_dir):
    """
    Generate and save all visualizations

    Args:
        model (torch.nn.Module): Trained model
        dataloaders (dict): Dictionary of dataloaders
        class_names (list): List of class names
        device (torch.device): Device to run the model on
        output_dir (str): Output directory
    """
    # Check if output_dir has redundant model name structure
    # For example: 'efficientnet_b0/visualizations/efficientnet_b0'
    path_parts = output_dir.split(os.sep)
    if len(path_parts) >= 3 and path_parts[-1] == path_parts[-3]:
        # Remove the redundant directory by moving up one level
        output_dir = os.path.dirname(output_dir)
        print(
            f"Detected redundant directory structure. Using output directory: {output_dir}"
        )

    os.makedirs(output_dir, exist_ok=True)

    # Collect true labels and predictions
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in dataloaders["test"]:
            inputs = inputs.to(device)
            labels = labels.cpu().numpy()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy()

            all_labels.extend(labels)
            all_preds.extend(preds)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # 1. Confusion Matrix
    fig = plot_confusion_matrix(all_labels, all_preds, class_names)
    fig.savefig(
        os.path.join(output_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    # 2. Classification Report
    fig = plot_classification_report(all_labels, all_preds, class_names)
    fig.savefig(
        os.path.join(output_dir, "classification_report.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 3. Class Distribution
    fig = plot_class_distribution(dataloaders["test"].dataset, class_names)
    fig.savefig(
        os.path.join(output_dir, "class_distribution.png"), dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    # 4. Example Predictions
    fig = visualize_model_predictions(model, dataloaders["test"], class_names, device)
    fig.savefig(
        os.path.join(output_dir, "example_predictions.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 5. Misclassified Examples
    fig = plot_examples_of_misclassifications(
        model, dataloaders["test"], class_names, device
    )
    fig.savefig(
        os.path.join(output_dir, "misclassified_examples.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"All visualizations saved to {output_dir}")


# Add CV2 import if GradCAM is being used
try:
    import cv2
except ImportError:
    print(
        "Warning: OpenCV (cv2) is required for GradCAM visualization. Install with 'pip install opencv-python'"
    )
