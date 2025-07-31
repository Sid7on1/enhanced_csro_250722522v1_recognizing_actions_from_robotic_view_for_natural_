import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import logging
import torchvision.transforms as transforms
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageDataset(Dataset):
    """
    Image Dataset for loading and transforming images.

    Args:
        data_df (pd.DataFrame): DataFrame containing image file paths and labels.
        transform (callable, optional): Optional transform to be applied on a sample.
        velocity_threshold (float): Velocity threshold for action recognition.
        flow_theory (bool): Whether to apply Flow Theory for motion estimation.

    Attributes:
        data_df (pd.DataFrame): DataFrame containing image file paths and labels.
        transform (callable, optional): Optional transform to be applied on a sample.
        velocity_threshold (float): Velocity threshold for action recognition.
        flow (np.ndarray): Optical flow matrix, computed using Flow Theory.
        image_files (list): List of image file paths.
        labels (list): List of labels for each image.
    """

    def __init__(self, data_df: pd.DataFrame, transform: Optional[callable] = None, velocity_threshold: float = 1.0, flow_theory: bool = True):
        self.data_df = data_df
        self.transform = transform
        self.velocity_threshold = velocity_threshold
        self.flow = None
        if flow_theory:
            self.compute_optical_flow()
        self.image_files = data_df['image_path'].tolist()
        self.labels = data_df['label'].tolist()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int):
        img_path = self.image_files[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        if self.flow is not None:
            flow_x = self.flow[idx, :, :, 0]
            flow_y = self.flow[idx, :, :, 1]
            return image, label, flow_x, flow_y
        else:
            return image, label

    def compute_optical_flow(self):
        """
        Compute optical flow using Flow Theory.
        """
        logger.info("Computing optical flow using Flow Theory...")
        # Placeholder implementation - replace with actual Flow Theory implementation
        # For now, generating random flow matrices
        height, width = 224, 224
        self.flow = np.random.rand(len(self.image_files), height, width, 2)

    def update_velocity_threshold(self, new_threshold: float):
        """
        Update the velocity threshold used for action recognition.

        Args:
            new_threshold (float): New velocity threshold value.
        """
        self.velocity_threshold = new_threshold

class DataLoader:
    """
    Data Loader for loading and batching image data.

    Args:
        dataset (Dataset): Image dataset to load data from.
        batch_size (int): Number of samples per batch.
        shuffle (bool, optional): Whether to shuffle the data after each epoch. Default is True.
        num_workers (int, optional): Number of worker processes for data loading. Default is 0.

    Attributes:
        dataset (Dataset): Image dataset to load data from.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data after each epoch.
        num_workers (int): Number of worker processes for data loading.
        data_loader (DataLoader): PyTorch DataLoader for batching and shuffling data.

    Raises:
        ValueError: If batch_size is less than or equal to 0.
    """

    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True, num_workers: int = 0):
        if batch_size <= 0:
            raise ValueError("Batch size must be greater than 0.")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def __iter__(self):
        """
        Return an iterator over the batches.
        """
        return iter(self.data_loader)

    def __len__(self):
        """
        Return the number of batches per epoch.
        """
        return len(self.data_loader)

    def update_dataset(self, new_dataset: Dataset):
        """
        Update the dataset used by the data loader.

        Args:
            new_dataset (Dataset): New dataset to use for data loading.
        """
        self.dataset = new_dataset
        self.data_loader.dataset = new_dataset  # Update the underlying dataset

    def batch_data(self):
        """
        Return the batched data as a generator.
        """
        return self.data_loader

    def set_shuffle(self, shuffle: bool):
        """
        Set the shuffle option for the data loader.

        Args:
            shuffle (bool): Whether to shuffle the data after each epoch.
        """
        self.shuffle = shuffle
        self.data_loader.shuffle = shuffle  # Update the shuffle option

    def set_num_workers(self, num_workers: int):
        """
        Set the number of worker processes for data loading.

        Args:
            num_workers (int): Number of worker processes.
        """
        self.num_workers = num_workers
        self.data_loader.num_workers = num_workers  # Update the number of workers

# Example usage
if __name__ == '__main__':
    # Placeholder data
    data = {
        'image_path': ['image1.jpg', 'image2.jpg', 'image3.jpg'],
        'label': ['action1', 'action2', 'action3']
    }
    data_df = pd.DataFrame(data)

    # Transforms for data augmentation
    transform = transforms.Compose([
        transforms.RandomCrop(224, pad_if_needed=True),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    # Create dataset and data loader
    dataset = ImageDataset(data_df, transform=transform)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)

    # Iterate over batches
    for batch in data_loader.batch_data():
        images, labels = batch
        print(images.shape)
        print(labels)