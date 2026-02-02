from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from .data_pipeline import DataPipeline
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, precision_score, recall_score, f1_score
import torch
from torch import nn, optim
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple, Any
import yaml
from config.logging_config import logger

class FocalLoss:
    """
    Focal Loss Implementation for Imbalanced Classification

    Focal loss addresses class imbalance by down-weighting easy examples and
    focusing training on hard negatives. Originally proposed in "Focal Loss for
    Dense Object Detection" (Lin et al., 2017).

    The focal loss formula is:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Where:
        - p_t: probability of the true class
        - alpha: class balancing weight (0-1)
        - gamma: focusing parameter (>=0), reduces loss for well-classified examples

    Runs hyperparameter tuning via Randomized Search to find the best alpha and gamma.
    Called by class ModelPipeline to run model with the best alpha and gamma.
    """
    def __init__(self, config, alpha=None, gamma=None):
        """
        Args:
            config (Dict[str, Any]): Configuration dictionary
            alpha (float, optional): Class balancing weight. Higher values weight positive class more.
            gamma (float, optional): Focusing parameter. Higher values down-weight easy examples more.
        """
        self.config = config
        self.model = None
        self.data_pipeline = DataPipeline(config)

        self.alpha = alpha if alpha is not None else config['best_alpha']
        self.gamma = gamma if gamma is not None else config['best_gamma']

        self.train_loader = self.data_pipeline.create_train_dataloader()
        self.test_loader = self.data_pipeline.create_test_dataloader()
        logger.info("Data loaders initialized.")

    def __call__(self, y_hat, y_true):
        """
        Makes the class instance callable like a function
        Only to be used by class ModelPipeline
        """
        return self.compute_focal_loss(y_hat, y_true, self.alpha, self.gamma)

    def focal_model(self):

        self.model = nn.Sequential(
            nn.Linear(29, 20),
            nn.ReLU(),
            nn.Linear(20, 12),
            nn.ReLU(),
            nn.Linear(12, 6),
            nn.ReLU(),
            nn.Linear(6, 3),
            nn.ReLU(),
            nn.Linear(3, 1)
            )
        return self.model

    def compute_focal_loss(self, y_hat, y_true, alpha, gamma):
        """
        Calculates focal loss for binary classification.

        The focal loss modifies cross-entropy by adding a modulating factor (1 - p_t)^gamma
        that reduces the loss contribution from easy examples and focuses learning on hard examples.

        Mathematical formulation:
            1. Compute BCE loss: BCE = -log(p) if y=1, -log(1-p) if y=0
            2. Get probability of true class: p_t = exp(-BCE)
            3. Compute focal weight: (1 - p_t)^gamma
            4. Apply class balancing: alpha_t = alpha if y=1 else (1-alpha)
            5. Final loss: FL = alpha_t * (1 - p_t)^gamma * BCE

        Args:
            y_hat: Raw model outputs (logits), shape (batch_size,)
            y_true: Ground truth labels (0 or 1), shape (batch_size,)
            alpha: Class balancing weight for positive class
            gamma: Focusing parameter

        Returns:
            Mean focal loss over the batch
        """
        # Compute binary cross entropy loss (element-wise, no reduction)
        bce_loss = F.binary_cross_entropy_with_logits(y_hat, y_true, reduction='none')

        # Compute probability of true class: p_t = exp(-BCE)
        # When BCE is low (good prediction), p_t is high
        # When BCE is high (bad prediction), p_t is low
        p_t = torch.exp(-bce_loss)

        # Compute focal weight: (1 - p_t)^gamma
        # Easy examples (high p_t) get low weight
        # Hard examples (low p_t) get high weight
        focal_weight = (1 - p_t) ** gamma

        # Apply class-specific alpha weighting
        # alpha for positive class, (1-alpha) for negative class
        alpha_t = torch.where(y_true == 1, alpha, 1 - alpha)
        focal_weight = alpha_t * focal_weight

        # Return mean focal loss
        return (focal_weight * bce_loss).mean()


    def focal_train(self, alpha, gamma, epochs=5):
        """
        Train model with specific alpha and gamma values
        ---
        Args:
            alpha (float): Alpha parameter for focal loss
            gamma (float): Gamma parameter for focal loss
            epochs (int): Number of training epochs
        Returns:
            Tuple[nn.Module, float]: Trained model and final loss
        """
        logger.info("Starting focal loss hyperparameter tuning ...")

        model = self.focal_model()
        optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])

        model.train()

        for epoch in range(epochs):
            epoch_loss = 0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                optimizer.zero_grad()

                output = model(data.float())
                loss = self.compute_focal_loss(output.squeeze(), target.float(), alpha, gamma)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            final_loss = epoch_loss / len(self.train_loader)

        return model, final_loss

    def focal_evaluate(self, model):
        """
        Evaluate model performance on test set
        ---
        Args:
            model (nn.Module): Trained model to evaluate
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, target in self.test_loader:
                output = model(data.float())
                predictions = torch.sigmoid(output.squeeze())

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        # Convert predictions to binary (threshold = 0.5)
        binary_predictions = (all_predictions > 0.5).astype(int)

        precision = precision_score(all_targets, binary_predictions, zero_division=0)
        recall = recall_score(all_targets, binary_predictions, zero_division=0)
        f1 = f1_score(all_targets, binary_predictions, zero_division=0)

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': all_predictions,
            'targets': all_targets
        }

    def focal_random_search(self, n_iterations=10):
        """
        Perform randomized hyperparameter search for optimal alpha and gamma.

        Args:
            n_iterations: Number of random combinations to try

        Returns:
            Tuple containing best parameters and results
        """
        alpha_grid = self.config['focal_loss_config']['alpha_grid']
        gamma_grid = self.config['focal_loss_config']['gamma_grid']
        priority_metric = self.config['priority_metric']

        logger.info(f"Starting Random Search with {n_iterations} out of {len(alpha_grid) * len(gamma_grid)} combinations, running 5 epochs per test. "
                f"{priority_metric} as decision metric.")

        results = []
        best_score = 0
        best_params = {}

        for i in range(n_iterations):
            # Randomly sample alpha and gamma
            alpha = random.choice(alpha_grid)
            gamma = random.choice(gamma_grid)

            logger.info(f"Currently testing combination {i+1}/{n_iterations}: "
                    f"[alpha {alpha}, gamma {gamma}] ...")

            trained_model, final_loss = self.focal_train(alpha, gamma, epochs=5)
            metrics = self.focal_evaluate(trained_model)

            current_score = metrics[priority_metric]  # Getting current score based on priority metric

            result = {
                'iteration': i+1,
                'alpha': alpha,
                'gamma': gamma,
                'final_loss': final_loss,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score']
            }
            results.append(result)

            if current_score > best_score:
                best_score = current_score
                best_params = {'alpha': alpha, 'gamma': gamma}
                best_model = trained_model

            logger.info(f"Best results for hyperparameter [alpha: {alpha}, gamma: {gamma}] | "
                    f"Precision: {metrics['precision']:.5f} | Recall: {metrics['recall']:.5f} | F1: {metrics['f1_score']:.5f}")

        logger.info("Random Search complete.")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best {priority_metric}: {best_score:.5f}")
