import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision import models

from PIL import Image
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from random import shuffle
import timm


class ImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform

        self.image_paths = []
        self.labels = []
        self.label_to_int = {}

        label_int = 0
        for label in os.listdir(directory):
            label_dir = os.path.join(directory, label)
            if os.path.isdir(label_dir):
                if label not in self.label_to_int:
                    self.label_to_int[label] = label_int
                    label_int += 1
                for image_name in os.listdir(label_dir):
                    self.image_paths.append(os.path.join(label_dir, image_name))
                    self.labels.append(self.label_to_int[label])
        print(self.label_to_int)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, shear=10, scale=(0.8,1.2)),
    transforms.RandomCrop((200, 200)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def accuracy(outputs, labels):
    """
    Compute the accuracy of the model.

    Parameters:
    outputs (torch.Tensor): The output predictions from the model.
    labels (torch.Tensor): The actual labels.

    Returns:
    float: The accuracy of the model.
    """
    # Get the index of the max log-probability (the predicted class)
    _, preds = torch.max(outputs, dim=1)
    # Calculate the number of correct predictions
    correct = (preds == labels).float()
    # Calculate the accuracy
    accuracy = correct.sum() / len(correct)
    return accuracy

def fit(epochs, model, loss_func, opt, train_dl, valid_dl, device):
    recorder = {'tr_loss': [], 'val_loss': [], 'tr_acc': [], 'val_acc': []}
    for epoch in range(epochs):
        model.train()
        train_loss, train_acc, train_count = 0., 0., 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_func(pred, yb)
            loss.backward()
            opt.step()
            opt.zero_grad()

            train_loss += loss.item() * len(xb)
            train_acc += accuracy(pred, yb).item() * len(xb)
            train_count += len(xb)

        recorder['tr_loss'].append(train_loss / train_count)
        recorder['tr_acc'].append(train_acc / train_count)

        model.eval()
        val_loss, val_acc, val_count = 0., 0., 0
        with torch.no_grad():
            for xb, yb in valid_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_func(pred, yb)

                val_loss += loss.item() * len(xb)
                val_acc += accuracy(pred, yb).item() * len(xb)
                val_count += len(xb)

        recorder['val_loss'].append(val_loss / val_count)
        recorder['val_acc'].append(val_acc / val_count)

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss / train_count:.4f}, "
              f"Acc: {train_acc / train_count:.4f}, Val Loss: {val_loss / val_count:.4f}, "
              f"Val Acc: {val_acc / val_count:.4f}")

    # Plotting outside the loop
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(recorder['tr_loss'], label='Train Loss', c='#983FFF', linestyle='-')
    ax.plot(recorder['val_loss'], label='Validation Loss', c='#FF9300', linestyle='--')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()

    return recorder


if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("Usage: train.py <path>")
        sys.exit(1)
    
    path = sys.argv[1]
    
    model = timm.create_model('convnext_small.fb_in22k', pretrained=True)

    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    
    dataset = ImageDataset(directory=path, transform=transforms)
    label2int = {'Apple_rust': 0, 'Apple_Black_rot': 1, 'Grape_Esca': 2, 'Apple_healthy': 3, 'Grape_healthy': 4, 'Grape_spot': 5, 'Apple_scab': 6, 'Grape_Black_rot': 7}
    int2label = {v:k for k,v in label2int.items()}

    # Determine the lengths of the splits
    train_len = int(0.8 * len(dataset))  # 80% of the dataset for training
    val_len = len(dataset) - train_len  # The rest for validation

    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    num_classes = 8
    model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)

    # Move the model to GPU if available
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f'>>>device: {device}')

    
    xb, yb = next(iter(train_loader))
    xb, yb = xb.to(device), yb.to(device)

    print(f'>>>xb shape: {xb.shape}')
    print(f'>>>yb shape: {yb.shape}')
    print(f'>>>xb mean: {xb.mean()}, xb std: {xb.std()}')

    batch_size=64
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.head.fc.parameters(), lr=0.005)
    epochs = 5
    
    recorder = fit(epochs, model, loss_fn, opt, train_loader, val_loader, device=device)
    
    print(f'saving model to {pathlib.Path.cwd()}')
    torch.save(model.state_dict(), 'model.pth')
    