import pandas as pd
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from config.logging_config import logger


class DataPipeline():
    def __init__(self, config):
        self.config = config
        self.dataset = None
        self.train_dataset = None
        self.test_dataset = None
        self.scaler = StandardScaler()

        # Automatically preprocess and split data
        self.preprocess_data()
        self.split_data()

    def preprocess_data(self):
        """
        Drops the target value and applies preprocessing to the data
        """
        df = pd.read_csv(self.config['data_path'])

        X = df.drop(columns=[self.config['target']])
        y = df[self.config['target']]

        X[['Amount']] = self.scaler.fit_transform(X[['Amount']])

        X_tensor = torch.FloatTensor(X.values)
        y_tensor = torch.FloatTensor(y.values)

        self.dataset = TensorDataset(X_tensor, y_tensor)
        return X_tensor.shape[1]

    def transform(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Preprocessing for MLP DataPipeline
        ---
        Returns:
            X_array, y_array (tuple): numpy arrays
        """
        df = pd.read_csv(self.config['data_path'])

        X = df.drop(columns=[self.config['target']])
        y = df[self.config['target']]

        X[['Amount']] = self.scaler.fit_transform(X[['Amount']])

        X_array = X.values.astype(np.float32)
        y_array = y.values.astype(np.float32)

        return X_array, y_array

    def split_data(self):
        """
        Splits the dataset using PyTorch
        """
        if self.config['random_state'] is not None:
            generator = torch.Generator().manual_seed(self.config['random_state'])
        else:
            generator = torch.Generator()

        exact_train_size = int(len(self.dataset) * self.config['train_size'])
        exact_test_size = len(self.dataset) - exact_train_size

        self.train_dataset, self.test_dataset = random_split(
            self.dataset, [exact_train_size, exact_test_size], generator=generator
        )

    def create_train_dataloader(self):
        """
        Create Training DataLoader for training dataset
        ---
        Args:
            batch_size (int): Batch size for training data loader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.config['train_batch_size'],
            shuffle=True,
            drop_last=False
        )

    def create_test_dataloader(self):
        """
        Create Test DataLoader for test dataset
        ---
        Args:
            batch_size (int): Batch size for test data loader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.config['test_batch_size'],
            shuffle=False,
            drop_last=False
        )

    def create_val_dataloader(self, val_ratio=0.20, batch_size=64):
        """
        Splitting validation set from training dataset
        Test dataset remains unchanged
        To be used in early stopping
        ---
        Args:
            val_ratio (float: Proportion of training data to use for validation)
        """

        train_size = int((1 - val_ratio) * len(self.train_dataset))
        val_size = len(self.train_dataset) - train_size

        generator = torch.Generator().manual_seed(self.config['random_state'])
        self.train_dataset, self.val_dataset = random_split(
            self.train_dataset,
            [train_size, val_size],
            generator=generator
        )

        logger.info(f"Validation split created - Train: {train_size}, Val: {val_size}")

        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle validation data
            drop_last=False
        )
