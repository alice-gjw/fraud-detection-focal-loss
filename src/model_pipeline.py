import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .data_pipeline import DataPipeline
from .focal_loss import FocalLoss
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, precision_score, recall_score, f1_score
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple, Any
import yaml
from config.logging_config import logger


class ModelPipeline:
    """
    End-to-end PyTorch model training and evaluation pipeline.

    Supports multiple model architectures, loss functions (including Focal Loss),
    and optimizers. Provides comprehensive training visualization and metrics tracking.
    """
    def __init__(self, config, model_name, auto_setup=True, **setup_kwargs) -> None:
        """
        Initialize trainer with configuration
        ---
        Args:
            config (Dict[str, Any]): Configuration dictionary
            model_name (str): Name for the model
        """
        self.config = config
        self.model_name = model_name
        self.data_pipeline = DataPipeline(config)

        self.input_features = config['input_features']

        self.model = None
        self.loss_fn = None
        self.optimizer = None

        # If early stopping used
        self.patience = None
        self.min_delta = None
        self.max_epochs = None

        # Setup configurations
        if auto_setup:  # auto_setup=True by default, will automatically call
            self._auto_setup(**setup_kwargs)

    def _auto_setup(self, **kwargs):
        """
        Automatically configure all components
        """
        self.configure_model(**kwargs.get('model', {}))
        self.setup_loss_fn(**kwargs.get('loss', {}))
        self.setup_optimizer(**kwargs.get('optimizer', {}))
        self.setup_dataloaders(**kwargs.get('data', {}))


    def configure_model(self,
                        layer_sizes=None,
                        initializer=None,
                        activ_fn=None,
                        activ_final=None,
                        architecture=None) -> nn.Sequential:
        """
        Builds a configured neural network
        ---
        Args:
            - layer_sizes (List[int]):
                List of layer sizes (default: [20, 12, 6, 3, 1])
            - activ_fn (nn.Module):
                Activation function for hidden layers (default: ReLU)
            - activ_final (nn.Module):
                Final layer activation (default: Sigmoid)
            - architecture (str):
                Specific changes to the model's architecture
                'dropout', 'batchnorm1d', None
        ---
        Returns:
            self.model (nn.Sequential)
        """

        logger.info(f"======= {self.model_name} =======")


        # LAYER SIZES
        if layer_sizes is not None:
            logger.info(f"Layer size: {[self.input_features] + layer_sizes}")

        layer_sizes = self.config['layer_sizes']
        logger.info(f"Layer size: {[self.input_features] + layer_sizes}")


        # INITIALIZER
        initializer_key = self.config['initializer']

        initializer_map = {
            'xavier_normal': nn.init.xavier_normal_,
            'xavier_uniform': nn.init.xavier_uniform_,
            'kaiming_normal': nn.init.kaiming_normal_,
            'kaiming_uniform': nn.init.kaiming_uniform_
        }

        if initializer_key not in initializer_map:
            raise ValueError(f"Unable to identify initializer from map.")

        initializer = initializer_map[initializer_key]
        logger.info(f"Initializer: {initializer.__name__}")


        # ACTIVATION FUNCTION
        activ_fn_key = self.config['activ_fn']

        activ_fn_map = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU()
        }

        if activ_fn_key not in activ_fn_map:
            raise ValueError(f"Unidentified Activation Function in config file.")

        activ_fn = activ_fn_map[activ_fn_key]
        logger.info(f"Activation Function: {activ_fn}")


        # FINAL ACTIVATION LAYER
        # Get in cases where there's 2 conditions, including None
        activ_final_key = self.config['activ_final']

        activ_final_map = {
            'null': None,
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'softmax': nn.Softmax(dim=1),
            'log_softmax': nn.LogSoftmax(dim=1)
        }

        if activ_final_key is None:
            activ_final = None
            logger.info("No final activation function layer")
        elif activ_final_key in activ_final_map:
            activ_final = activ_final_map[activ_final_key]
            logger.info(f"Final Activation Function: {activ_final}")


        # BUILDING LAYERS
        layers = []
        prev_size = self.input_features
        architecture = self.config['architecture']

        logger.info("========== Building model ===========")

        for curr_size in layer_sizes[:-1]: # All except the last layer
            logger.info(f"nn.Linear({prev_size}, {curr_size})")
            input_layer = nn.Linear(prev_size, curr_size)
            initializer(input_layer.weight)  # Apply weight initialization
            layers.append(input_layer) # Input layer

            if architecture == 'batchnorm':
                logger.info(f"nn.BatchNorm1d({curr_size})")
                layers.append(nn.BatchNorm1d(curr_size))

            logger.info(f"Activation: {type(activ_fn).__name__}")
            layers.append(activ_fn)

            if architecture == 'dropout':
                logger.info(f"Layer: nn.Dropout({self.config['dropout_rate']})")
                layers.append(nn.Dropout(self.config['dropout_rate']))

            prev_size = curr_size

        logger.info(f"Final layer: nn.Linear({prev_size}, {layer_sizes[-1]})")
        output_layer = nn.Linear(prev_size, layer_sizes[-1])
        initializer(output_layer.weight)
        layers.append(output_layer) # Add final layer

        # Only add final activation if one is specified
        if activ_final is not None:
            logger.info(f"Final activation layer: {type(activ_final).__name__}")
            layers.append(activ_final)

        logger.info("=====================================")

        self.model = nn.Sequential(*layers)
        return self.model

    # LOSS FUNCTION
    def setup_loss_fn(self, loss_fn=None) -> Union[nn.Module, str]:
        """
        Sets up the loss function.

        Supports:
        - BCELoss: Standard binary cross-entropy
        - BCEWithLogitsLoss: BCE with built-in sigmoid (more numerically stable)
        - ClassWeights: Weighted BCE for class imbalance
        - FocalLoss: Custom focal loss for severe class imbalance
        ---
        Args:
            loss_fn: Specifies the loss function
        ---
        Returns:
            self.loss_fn (Union[nn.Module, str])
        """
        loss_fn_key = self.config['loss_fn']

        # Validate loss function key first
        valid_loss_functions = {'bceloss',
                                'bcewithlogitsloss',
                                'classweights',
                                'focalloss'
                            }
        if loss_fn_key not in valid_loss_functions:
            raise ValueError("Unknown loss function.")

        if loss_fn_key == 'bceloss':
            self.loss_fn = nn.BCELoss()
            loss_name = "BCE Loss"

        elif loss_fn_key == 'bcewithlogitsloss':
            self.loss_fn = nn.BCEWithLogitsLoss()
            loss_name = "BCE With Logits Loss"

        elif loss_fn_key == 'classweights':
            class_weights = torch.tensor(self.config['class_weights'], dtype=torch.float32)
            pos_weight = class_weights[1] / class_weights[0]
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss_name = f"Class Weighted BCE Loss, Class Weights: {self.config['class_weights']}"

        elif loss_fn_key == 'focalloss':
            def focal_loss(y_hat, y):
                """
                Inline focal loss function for use in training loop.

                Focal Loss = -alpha_t * (1 - p_t)^gamma * BCE

                This implementation:
                1. Converts logits to probabilities via sigmoid
                2. Computes p_t (probability of true class)
                3. Applies focal modulation: (1 - p_t)^gamma
                4. Applies class balancing via alpha
                """
                alpha = self.config['alpha']
                gamma = self.config['gamma']

                # Get probabilities from logits
                p = torch.sigmoid(y_hat)

                # Calculate p_t (probability of true class)
                # If y=1, p_t = p; if y=0, p_t = 1-p
                p_t = torch.where(y == 1, p, 1 - p)

                # Calculate focal weight: (1 - p_t)^gamma
                # Easy examples (high p_t) get low weight
                focal_weight = (1 - p_t) ** gamma

                # Calculate alpha_t (class weighting)
                # alpha for positive, (1-alpha) for negative
                alpha_t = torch.where(y == 1, alpha, 1 - alpha)

                # Calculate binary cross entropy manually
                # BCE = -[y*log(p) + (1-y)*log(1-p)]
                bce_loss = -(y * torch.log(p + 1e-8) + (1 - y) * torch.log(1 - p + 1e-8))

                # Apply focal loss formula
                focal_loss = alpha_t * focal_weight * bce_loss

                return focal_loss.mean()

            self.loss_fn = focal_loss
            loss_name = "Focal Loss"

        logger.info(f"Loss Function: {loss_name}")
        return self.loss_fn


    # OPTIMIZER
    def setup_optimizer(self, learning_rate=None, optimizer=None) -> torch.optim.Optimizer:
        """
        Sets up the optimizer
        ---
        Args:
            learning_rate: (default: 0.001)
            optimizer: Choice of which optimizer to use  (default: Adam)
        ---
        Returns:
            self.optimizer (torch.optim.Optimizer)
        """
        # LEARNING RATE
        if learning_rate is not None:
            logger.info(f"Learning Rate: {learning_rate}")

        learning_rate = self.config['learning_rate']
        logger.info(f"Learning rate: {learning_rate}")


        # OPTIMIZER
        optimizer_key = self.config['optimizer']

        optimizer_map = {
            'adam': optim.Adam,
            'sgd': optim.SGD,
            'rmsprop': optim.RMSprop
        }

        if optimizer_key not in optimizer_map:
            raise ValueError(f"Unable to identify Optimizer from the map.")

        optimizer = optimizer_map[optimizer_key]
        logger.info(f"Optimizer: {optimizer.__name__}")

        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        return self.optimizer

    # DATALOADERS
    def setup_dataloaders(self,
                        train_batch_size=None,
                        test_batch_size=None) -> None:
        """
        Sets up training and testing data loaders from class DataPipeline
        ---
        Args:
            train_batch_size (int):
            test_batch_size (int):
        """
        logger.info(f"Train Batch Size: {self.config['train_batch_size']}")
        logger.info(f"Test Batch Size: {self.config['test_batch_size']}")

        self.train_loader = self.data_pipeline.create_train_dataloader()
        self.test_loader = self.data_pipeline.create_test_dataloader()


    # PLOTTING GRAPHS
    def plot_graphs(self, train_loss, test_loss, precision_scores, recall_scores,
                        f1_scores, y_pred, y_true, y_proba) -> None:
        """
        Plot all three training visualization graphs in one function

        Args:
            - train_loss (List[float]):
                List of training losses per epoch
            - test_loss (List[float]):
                List of testing losses per epoch
            - precision_scores (List[float]):
                List of precision scores per epoch
            - recall_scores (List[float]):
                List of recall scores per epoch
            - f1_scores (List[float]):
                List of F1 scores per epoch
            - y_true (List[float]):
                True binary labels (from final epoch)
            - y_proba (List[float]):
                Predicted probabilities for the positive class (from final epoch)
        """

        # Plotting confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)

        df_cm = pd.DataFrame(conf_matrix,
                            index=['No Fraud', 'Fraud'],
                            columns=['No Fraud', 'Fraud'])
        logger.info("==== Confusion Matrix ====")
        logger.info(f"\n{df_cm}")
        logger.info("Total samples: 284807")

        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 18))

        # Precision-Recall vs Threshold
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        axes[0].plot(thresholds, precision[:-1], 'b-', label='Precision', linewidth=2)
        axes[0].plot(thresholds, recall[:-1], 'r-', label='Recall', linewidth=2)
        axes[0].set_xlabel('Threshold')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Precision-Recall vs Threshold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim([0, 1])
        axes[0].set_ylim([0, 1])

        # Training vs Testing Loss
        epochs = range(1, len(train_loss) + 1)
        axes[1].plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
        axes[1].plot(epochs, test_loss, 'r-', label='Testing Loss', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Training vs Testing Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Precision, Recall & F1 Score Over Epochs
        axes[2].plot(epochs, precision_scores, 'b-', label='Precision', linewidth=2, marker='o')
        axes[2].plot(epochs, recall_scores, 'r-', label='Recall', linewidth=2, marker='s')
        axes[2].plot(epochs, f1_scores, 'g-', label='F1 Score', linewidth=2, marker='^')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Score')
        axes[2].set_title('Precision, Recall & F1 Score Over Epochs')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim([0, 1])

        plt.tight_layout()
        plt.show()


    def train_and_evaluate(
        self,
        epochs=None
    ) -> Tuple[List[float], List[float], List[float], List[float], List[float], float, int]:
        """
        Trains the model with forward pass, evaluates the model
        Prints metrics for each epoch and stores metrics for plotting
        ---
        Args:
            epochs:
        ---
        Returns:
            Tuple[List[float], List[float], List[float], List[float], List[float], float, int]
            A tuple containing:
                - train_loss (List[float]):
                    Average training loss for each epoch
                - test_losses (List[float]):
                    Average test/validation loss for each epoch
                - precision_scores (List[float]):
                    Precision scores for the positive class (fraud) for each epoch
                - recall_scores (List[float]):
                    Recall scores for the positive class (fraud) for each epoch
                - f1_scores (List[float]):
                    F1-scores for the positive class (fraud) for each epoch
                - best_f1 (float):
                    Highest F1-score achieved during training
                - best_epoch (int):
                    Epoch number where the best F1-score was achieved
        """
        if self.model is None:
            raise ValueError("Your model has not been created. Call train_and_evaluate() first)")

        if epochs is None:
            epochs = self.config['epochs']

        # Initialize (give it the initial value) before training starts
        # Will reset to 0 each epoch if within the loop
        best_f1 = 0.0
        best_epoch = 0

        train_loss = []
        test_loss = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        logger.info(f"Probability threshold set to {self.config['threshold']}.")
        logger.info("Beginning training and evaluation.")

        # Loop for training and evaluation
        for epoch_idx in range(epochs):

            self.model.train()  # 1. Training phase
            epoch_sum_train_loss = []

            for batch in self.train_loader: # Using train dataset
                x, y = batch
                y_hat = self.model(x).squeeze()  # Forward pass, obtaining predictions
                batch_train_loss = self.loss_fn(y_hat, y)
                epoch_sum_train_loss.append(batch_train_loss.item())

                batch_train_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.model.eval()  # 2. Evaluation phase
            all_predictions = []
            all_targets = []
            all_probabilities = []
            epoch_sum_test_loss = []

            with torch.no_grad(): # No gradient computation needed
                for batch in self.test_loader:  # Using test dataset
                    x, y = batch
                    y_hat = self.model(x).squeeze()
                    batch_test_loss = self.loss_fn(y_hat, y)
                    epoch_sum_test_loss.append(batch_test_loss.item())

                    probabilities = torch.sigmoid(y_hat)
                    predictions = (probabilities > self.config['threshold']).float()

                    all_predictions.extend(predictions.cpu().numpy())  # .extend() lists it individually vs .append() adds entire object as one element
                    all_targets.extend(y.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())

            # 3. Average loss for the epoch
            avg_train_loss = sum(epoch_sum_train_loss) / len(epoch_sum_train_loss)
            avg_test_loss = sum(epoch_sum_test_loss) / len(epoch_sum_test_loss)

            # 4. Convert to numpy arrays for classification report
            y_true = np.array(all_targets)
            y_pred = np.array(all_predictions)

            report_dict = classification_report(y_true, y_pred,
                                                target_names=['No Fraud (0)', 'Fraud (1)'],
                                                output_dict=True,
                                                zero_division=0)


            precision = report_dict['Fraud (1)']['precision']
            recall = report_dict['Fraud (1)']['recall']
            f1 = report_dict['Fraud (1)']['f1-score']

            # Check if this is the best F1 score
            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch_idx + 1

            # Print results for this epoch
            logger.info(f"Epoch {epoch_idx + 1}/{self.config['epochs']} completed. "
                f"Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | "
                f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

            # Store metrics for plotting
            train_loss.append(avg_train_loss)
            test_loss.append(avg_test_loss)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        # End loop
        logger.info(f"Best F1 score: {best_f1:.4f} at Epoch {best_epoch}")

        self.plot_graphs(
            train_loss=train_loss,
            test_loss=test_loss,
            precision_scores=precision_scores,
            recall_scores=recall_scores,
            f1_scores=f1_scores,
            y_pred=y_pred,
            y_true=all_targets,  # From last epoch
            y_proba=all_probabilities  # From last epoch
        )
