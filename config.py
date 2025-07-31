import os
import logging
from typing import Dict, List, Tuple
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.transforms import transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    """
    Configuration class for the model and training.

    This class provides a centralized place to store and access various configuration settings
    for the model and training process. It also includes methods for loading and saving
    configurations to disk.
    """

    def __init__(self):
        self.model_name = "action_recognition_model"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = 10
        self.input_size = (224, 224)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.batch_size = 32
        self.num_workers = 4
        self.pin_memory = True
        self.shuffle = True
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        self.data_dir = "path/to/dataset"
        self.checkpoint_dir = "checkpoints"
        self.resume_checkpoint = None
        self.start_epoch = 0
        self.num_epochs = 50
        self.learning_rate = 0.001
        self.momentum = 0.9
        self.weight_decay = 0.0005
        self.milestones = [30, 40]
        self.gamma = 0.1
        self.log_interval = 10
        self.checkpoint_interval = 5

    def to_dict(self) -> Dict:
        """Convert the config object to a dictionary."""
        return self.__dict__

    def load_from_dict(self, config_dict: Dict) -> None:
        """Load the config object from a dictionary."""
        self.__dict__.update(config_dict)

    def save(self, filename: str) -> None:
        """Save the configuration to a file."""
        with open(filename, "w") as f:
            config_dict = self.to_dict()
            f.write(str(config_dict))
        logger.info(f"Configuration saved to {filename}")

    def load(self, filename: str) -> None:
        """Load the configuration from a file."""
        if not os.path.exists(filename):
            logger.error(f"Configuration file {filename} does not exist.")
            return
        with open(filename, "r") as f:
            config_dict = eval(f.read())  # Evaluate the string representation back to a dict
        self.load_from_dict(config_dict)
        logger.info(f"Configuration loaded from {filename}")

class ModelConfig:
    """
    Configuration class for the model architecture.

    This class provides a set of configuration settings specific to the model architecture,
    including the choice of backbone network, number of output classes, and other
    architecture-related parameters.
    """

    def __init__(self):
        self.backbone = "resnet50"
        self.num_classes = 10
        self.pretrained = True
        self.freeze_backbone = False

class DataConfig:
    """
    Configuration class for the data loading and preprocessing.

    This class includes settings related to the data loading process, such as dataset paths,
    batch size, data transformations, and worker configurations.
    """

    def __init__(self):
        self.data_dir = "path/to/dataset"
        self.batch_size = 32
        self.num_workers = 4
        self.pin_memory = True
        self.shuffle = True
        self.transform = None  # To be defined based on the specific dataset
        self.mean = None
        self.std = None

class OptimizerConfig:
    """
    Configuration class for the optimizer.

    This class includes settings related to the optimization process, such as the choice of
    optimizer, learning rate, momentum, weight decay, and learning rate scheduler.
    """

    def __init__(self):
        self.optimizer = "adam"
        self.learning_rate = 0.001
        self.momentum = 0.9
        self.weight_decay = 0.0005
        self.milestones = None
        self.gamma = 0.1

class TrainingConfig:
    """
    Configuration class for the training process.

    This class includes settings specific to the training process, such as the number of epochs,
    logging interval, checkpoint saving interval, and other training-related parameters.
    """

    def __init__(self):
        self.num_epochs = 50
        self.log_interval = 10
        self.checkpoint_interval = 5

# Example usage
if __name__ == "__main__":
    # Create configuration objects
    config = Config()
    model_config = ModelConfig()
    data_config = DataConfig()
    optimizer_config = OptimizerConfig()
    training_config = TrainingConfig()

    # Print configuration settings
    logger.info("Configuration Settings:")
    logger.info(f"Model Name: {config.model_name}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Number of Classes: {model_config.num_classes}")
    logger.info(f"Input Size: {config.input_size}")
    logger.info(f"Mean: {config.mean}")
    logger.info(f"Standard Deviation: {config.std}")
    logger.info(f"Batch Size: {data_config.batch_size}")
    logger.info(f"Learning Rate: {optimizer_config.learning_rate}")
    logger.info(f"Number of Epochs: {training_config.num_epochs}")

    # Save and load configurations
    config.save("config.txt")
    new_config = Config()
    new_config.load("config.txt")
    logger.info("Loaded Configuration:")
    logger.info(new_config.to_dict())