import os
import logging
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()

# Define constants and configuration
CONFIG = {
    'DATA_DIR': 'data',
    'MODEL_DIR': 'models',
    'LOG_DIR': 'logs',
    'BATCH_SIZE': 32,
    'EPOCHS': 10,
    'LEARNING_RATE': 0.001,
    'WEIGHT_DECAY': 0.0005,
    'CLASS_WEIGHTS': None,
    'SCALER': StandardScaler(),
    'DEVICE': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

# Define exception classes
class DataError(Exception):
    """Raised when data is invalid or missing"""
    pass

class ModelError(Exception):
    """Raised when model is invalid or missing"""
    pass

class TrainingError(Exception):
    """Raised when training fails"""
    pass

# Define data structures and models
class ActionDataset(Dataset):
    """Action dataset class"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []
        self.labels = []
        for file in os.listdir(data_dir):
            if file.endswith('.jpg'):
                self.data.append(os.path.join(data_dir, file))
                self.labels.append(int(file.split('_')[0]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.data[idx])
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

class ActionModel(nn.Module):
    """Action model class"""
    def __init__(self):
        super(ActionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define utility functions
def load_data(data_dir):
    """Load data from directory"""
    try:
        data = pd.read_csv(os.path.join(data_dir, 'data.csv'))
        return data
    except FileNotFoundError:
        raise DataError('Data file not found')

def preprocess_data(data):
    """Preprocess data"""
    scaler = CONFIG['SCALER']
    data['features'] = scaler.fit_transform(data['features'])
    return data

def create_dataset(data):
    """Create dataset"""
    dataset = ActionDataset(data_dir=CONFIG['DATA_DIR'])
    return dataset

def create_dataloader(dataset):
    """Create dataloader"""
    dataloader = DataLoader(dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True)
    return dataloader

def train_model(model, dataloader):
    """Train model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['LEARNING_RATE'], weight_decay=CONFIG['WEIGHT_DECAY'])
    for epoch in range(CONFIG['EPOCHS']):
        model.train()
        total_loss = 0
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(CONFIG['DEVICE']), labels.to(CONFIG['DEVICE'])
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(f'Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}')
    return model

def evaluate_model(model, dataloader):
    """Evaluate model"""
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(CONFIG['DEVICE']), labels.to(CONFIG['DEVICE'])
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
    accuracy = total_correct / len(dataloader.dataset)
    logger.info(f'Accuracy: {accuracy:.4f}')
    return accuracy

def save_model(model):
    """Save model"""
    torch.save(model.state_dict(), os.path.join(CONFIG['MODEL_DIR'], 'model.pth'))

def load_model():
    """Load model"""
    try:
        model = ActionModel()
        model.load_state_dict(torch.load(os.path.join(CONFIG['MODEL_DIR'], 'model.pth')))
        return model
    except FileNotFoundError:
        raise ModelError('Model file not found')

# Define main training function
def train_pipeline():
    """Training pipeline"""
    try:
        data = load_data(CONFIG['DATA_DIR'])
        data = preprocess_data(data)
        dataset = create_dataset(data)
        dataloader = create_dataloader(dataset)
        model = ActionModel()
        model = train_model(model, dataloader)
        accuracy = evaluate_model(model, dataloader)
        save_model(model)
        logger.info(f'Training complete, accuracy: {accuracy:.4f}')
    except DataError as e:
        logger.error(f'Data error: {e}')
    except ModelError as e:
        logger.error(f'Model error: {e}')
    except TrainingError as e:
        logger.error(f'Training error: {e}')

# Define command-line interface
def main():
    parser = argparse.ArgumentParser(description='Training pipeline')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--model_dir', type=str, default='models', help='Model directory')
    parser.add_argument('--log_dir', type=str, default='logs', help='Log directory')
    args = parser.parse_args()
    CONFIG['DATA_DIR'] = args.data_dir
    CONFIG['MODEL_DIR'] = args.model_dir
    CONFIG['LOG_DIR'] = args.log_dir
    train_pipeline()

if __name__ == '__main__':
    main()