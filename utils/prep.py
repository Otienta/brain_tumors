import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_pytorch_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, test_transform

def get_pytorch_dataloaders(data_dir='dataset'):
    train_transform, test_transform = get_pytorch_transforms()
    train_dataset = datasets.ImageFolder(f'{data_dir}/training', transform=train_transform)
    test_dataset = datasets.ImageFolder(f'{data_dir}/testing', transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
    return train_loader, test_loader, train_dataset.classes

def get_tensorflow_generators(data_dir='dataset'):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        f'{data_dir}/training',
        target_size=(224, 224),
        batch_size=16,
        class_mode='sparse'
    )
    test_generator = test_datagen.flow_from_directory(
        f'{data_dir}/testing',
        target_size=(224, 224),
        batch_size=16,
        class_mode='sparse'
    )
    return train_generator, test_generator, train_generator.class_indices