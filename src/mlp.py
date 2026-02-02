import numpy as np

class MLPTwoLayers:
    """
    A two-layer Multi-Layer Perceptron (MLP) neural network built from scratch with NumPy.

    This implementation demonstrates the core concepts of neural networks without
    relying on deep learning frameworks. It includes:
    - Forward propagation with sigmoid and softmax activations
    - Cross-entropy loss computation
    - Backpropagation with gradient descent
    - He weight initialization for better convergence

    Architecture:
    Input Layer -> Hidden Layer (Sigmoid) -> Output Layer (Softmax)
    """

    def __init__(self, input_size=3072, hidden_size=100, output_size=10):
        """
        Initialize the MLP with He weight initialization.

        He initialization helps prevent vanishing/exploding gradients by scaling
        weights based on the number of input neurons: sqrt(2.0 / input_size)
        ---
        Args:
            input_size : int, default=3072
                Number of input features
            hidden_size : int, default=100
                Number of neurons in the hidden layer
            output_size : int, default=10
                Number of output neurons (classes)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # He initialization for weights (optimal for ReLU-like activations)
        # Creating input_size x hidden_size matrix with He scaling
        self.w1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2.0 / self.input_size)

        # Biases for hidden layer (b1) - initialized to zeros
        # Shape: (1, hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))

        # Weights for hidden to output layer (w2) with He scaling
        # Shape: (hidden_size, output_size)
        self.w2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2.0 / self.hidden_size)

        # Biases for output layer (b2)
        # Shape: (1, output_size)
        self.b2 = np.zeros((1, self.output_size))

        # Learning rate for gradient descent
        self.learning_rate = 0.001

    def forward(self, features):
        """
        Forward propagation through the network.

        Computes:
        1. Hidden layer: z1 = X·W1 + b1, a1 = sigmoid(z1)
        2. Output layer: z2 = a1·W2 + b2, a2 = softmax(z2)
        ---
        Args:
            features : numpy.ndarray
                Input data of shape (batch_size, input_size)
        ---
        Returns:
            predictions : numpy.ndarray
                Network output probabilities of shape (batch_size, output_size)
        """
        # Store inputs for backpropagation
        self.X = features

        # Forward pass to hidden layer
        # z1 = X·W1 + b1 (linear transformation)
        self.z1 = np.dot(features, self.w1) + self.b1
        # a1 = sigmoid(z1) (non-linear activation)
        self.a1 = self._sigmoid(self.z1)

        # Forward pass to output layer
        # z2 = a1·W2 + b2 (linear transformation)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        # a2 = softmax(z2) (probability distribution over classes)
        self.a2 = self._softmax(self.z2)

        return self.a2

    def loss(self, predictions, label):
        """
        Compute cross-entropy loss.

        Cross-entropy loss measures the difference between predicted probability
        distribution and true distribution: L = -sum(y_true * log(y_pred))
        ---
        Args:
            predictions : numpy.ndarray
                Network predictions of shape (batch_size, output_size)
            label : numpy.ndarray or int or float
                True labels, can be one-hot encoded, class indices, or single class index
        ---
        Returns:
            loss : float
                Cross-entropy loss value
        """
        # Handle single sample case - convert to array if needed
        if np.isscalar(label):
            label = np.array([int(label)])
        elif isinstance(label, (int, float, np.integer, np.floating)):
            label = np.array([int(label)])
        elif hasattr(label, 'ndim') and label.ndim == 0:  # 0-dimensional array
            label = np.array([int(label.item())])

        # Ensure predictions is 2D
        if predictions.ndim == 1:
            predictions = predictions.reshape(1, -1)

        # Convert labels to one-hot encoding if they are class indices
        if label.ndim == 1 and len(label) == 1:
            num_samples = 1  # Single class index case
            one_hot_labels = np.zeros((num_samples, self.output_size))
            one_hot_labels[0, int(label[0])] = 1
            label = one_hot_labels

        elif label.ndim == 1 and len(label) > 1 and not np.all((label >= 0) & (label <= 1)):
            num_samples = label.shape[0]   # Multiple class indices case
            one_hot_labels = np.zeros((num_samples, self.output_size))
            one_hot_labels[np.arange(num_samples), label.astype(int)] = 1
            label = one_hot_labels

        elif label.ndim == 1 and len(label) != self.output_size:
            # If it's 1D but not the right size for one-hot, treat as class indices
            num_samples = label.shape[0]
            one_hot_labels = np.zeros((num_samples, self.output_size))
            one_hot_labels[np.arange(num_samples), label.astype(int)] = 1
            label = one_hot_labels

        elif label.ndim == 1:
            # If it's already one-hot but 1D, reshape it
            label = label.reshape(1, -1)

        # Store labels for backpropagation
        self.y = label

        # Compute cross-entropy loss
        # Add small epsilon to prevent log(0) which would give -inf
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)

        # Cross-entropy: L = -mean(sum(y * log(pred)))
        loss = -np.mean(np.sum(label * np.log(predictions), axis=1))

        return loss

    def backward(self):
        """
        Backpropagation to compute gradients and update weights.

        Uses the chain rule to propagate gradients backward:
        1. Output layer: dL/dW2 = a1.T · (a2 - y)
        2. Hidden layer: dL/dW1 = X.T · (dz2·W2.T * sigmoid'(z1))

        Then updates weights using gradient descent:
        W = W - learning_rate * gradient
        """
        batch_size = self.X.shape[0]

        # ===== Output Layer Gradients =====
        # For softmax + cross-entropy, the gradient simplifies to: dL/dz2 = a2 - y
        dz2 = self.a2 - self.y

        # Gradient for W2: dL/dW2 = a1.T · dz2
        dw2 = np.dot(self.a1.T, dz2) / batch_size

        # Gradient for b2: mean of dz2 across batch
        db2 = np.mean(dz2, axis=0, keepdims=True)

        # ===== Hidden Layer Gradients =====
        # Propagate gradient back through W2
        da1 = np.dot(dz2, self.w2.T)

        # Apply sigmoid derivative: dL/dz1 = da1 * sigmoid'(z1)
        dz1 = da1 * self._sigmoid_derivative(self.z1)

        # Gradient for W1: dL/dW1 = X.T · dz1
        dw1 = np.dot(self.X.T, dz1) / batch_size

        # Gradient for b1
        db1 = np.mean(dz1, axis=0, keepdims=True)

        # ===== Update weights and biases using gradient descent =====
        self.w2 -= self.learning_rate * dw2
        self.b2 -= self.learning_rate * db2
        self.w1 -= self.learning_rate * dw1
        self.b1 -= self.learning_rate * db1

    def _sigmoid(self, z):
        """
        Sigmoid activation function: sigma(z) = 1 / (1 + exp(-z))

        Squashes input values to range (0, 1).
        Used in hidden layer to introduce non-linearity.
        """
        # Clip z to prevent overflow in exp
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))


    def _sigmoid_derivative(self, z):
        """
        Derivative of sigmoid function: sigma'(z) = sigma(z) * (1 - sigma(z))

        Measures how sensitive the sigmoid output is to changes in input.
        Maximum gradient (0.25) occurs at z=0, approaches 0 at extremes.
        """
        s = self._sigmoid(z)
        return s * (1 - s)

    def _softmax(self, z):
        """
        Softmax activation function: softmax(z)_i = exp(z_i) / sum(exp(z_j))

        Converts raw scores (logits) into a probability distribution.
        Output values sum to 1 and are all positive.
        """
        # Subtract max for numerical stability (prevents overflow)
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def predict(self, features):
        """
        Make predictions on new data.
        ---
        Args:
            features : numpy.ndarray
                Input data of shape (batch_size, input_size)
        ---
        Returns:
            predictions : numpy.ndarray
                Predicted class indices of shape (batch_size,)
        """
        probabilities = self.forward(features)
        return np.argmax(probabilities, axis=1)

    def get_weights(self):
        """
        Return current weights and biases.
        ---
        Returns:
            dict : Dictionary containing all weights and biases
        """
        return {
            'w1': self.w1,
            'b1': self.b1,
            'w2': self.w2,
            'b2': self.b2
        }

    def set_learning_rate(self, lr):
        """
        Set the learning rate for training.
        """
        self.learning_rate = lr
