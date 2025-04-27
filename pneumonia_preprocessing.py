import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import torchvision


# Define transformations for the images (resize, convert to tensor, normalize)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Set the path to your dataset
data_dir = 'chest_xray'

# Load the original train dataset and apply train transforms
full_train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)

# Split into training and validation sets (e.g., 80% train, 20% val)
val_size = int(0.2 * len(full_train_dataset))
train_size = len(full_train_dataset) - val_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# Apply different transform to validation dataset
val_dataset.dataset.transform = val_test_transform

# Load test dataset
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=val_test_transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Train images: {len(train_dataset)}, Validation images: {len(val_dataset)}, Test images: {len(test_dataset)}")
print(full_train_dataset.class_to_idx)  # Should show {'NORMAL': 0, 'PNEUMONIA': 1}

# Function to unnormalize and show image
def imshow(img):
    img = img * 0.5 + 0.5  # Unnormalize from [-1, 1] to [0, 1]
    npimg = img.numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.axis('off')
    plt.show()

# Show sample training images
dataiter = iter(train_loader)
images, labels = next(dataiter)
print("Sample images from training set after augmentation + normalization:")
imshow(torchvision.utils.make_grid(images[:8]))  # Show first 8 images
