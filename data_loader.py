import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Set this to your actual dataset path
DATASET_DIR = "dataset/"

# Image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

def get_dataloaders(batch_size=16, val_split=0.2):
    full_dataset = datasets.ImageFolder(root=DATASET_DIR, transform=transform)

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, full_dataset.classes
