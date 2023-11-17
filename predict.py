from PIL import Image
import sys
import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import timm


def accuracy(outputs, labels):
    """
    Compute the accuracy of the model.

    Parameters:
    outputs (torch.Tensor): The output predictions from the model.
    labels (torch.Tensor): The actual labels.

    Returns:
    float: The accuracy of the model.
    """
    
    _, preds = torch.max(outputs, dim=1)
    correct = (preds == labels).float()
    accuracy = correct.sum() / len(correct)
    return accuracy


def predict(model, image_path, device, labels):
    img = Image.open(image_path)
    model = model.eval()

    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    output = model(transforms(img).unsqueeze(0).to(device))

    top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)
    pred_idx = top5_class_indices[0,0].item()
    print(f'Predicted: {labels[pred_idx]}, idx: {pred_idx}')
    return labels[pred_idx], transforms(img).numpy().transpose(1,2,0)
    

def main():

    if len(sys.argv) != 2:
        print("Usage: predict.py <path>")
        sys.exit(1)
    image_path = sys.argv[1]
    
    label2int = {
        'Apple_rust': 0,
        'Apple_Black_rot': 1,
        'Grape_Esca': 2,
        'Apple_healthy': 3,
        'Grape_healthy': 4,
        'Grape_spot': 5,
        'Apple_scab': 6,
        'Grape_Black_rot': 7
        }
    int2label = {v:k for k,v in label2int.items()}
    num_classes = 8
    
    model = timm.create_model('convnext_small.fb_in22k', pretrained=True, num_classes=num_classes)

    # Move the model to GPU if available
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f'>>>device: {device}')

    prediction, preprocessed_img = predict(model, image_path=image_path, device=device, labels=int2label)
    
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Display original image
    axes[0].imshow(Image.open(image_path))
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Display transformed/preprocessed image
    axes[1].imshow(preprocessed_img.astype('uint8'))
    axes[1].set_title("Transformed Image")
    axes[1].axis('off')

    # Add classification label
    fig.text(0.5, 0.01,
             f"DL classification\nClass predicted: {prediction}",
             ha="center",
             fontsize=12,
             bbox={"facecolor":"green", "alpha":0.3, "pad":3})

    plt.tight_layout(pad=3.0)
    plt.show()

if __name__ == "__main__":
    main()
