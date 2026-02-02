# Credit Card Fraud Detection with Focal Loss

A deep learning project for detecting fraudulent credit card transactions using neural networks with **Focal Loss** optimization for handling extreme class imbalance.

## Project Overview

Credit card fraud detection presents a challenging machine learning problem due to severe class imbalance (typically <0.2% fraud cases). This project implements:

1. **PyTorch-based neural network pipeline** with configurable architectures
2. **Focal Loss implementation** for handling class imbalance
3. **MLP from scratch** using only NumPy to demonstrate core neural network concepts
4. **Hyperparameter tuning** via randomized search

## Key Features

- **Focal Loss**: Custom loss function that down-weights easy examples and focuses on hard-to-classify cases
- **Modular Design**: Separate classes for data preprocessing, model training, and evaluation
- **Configuration-Driven**: YAML-based configuration for all hyperparameters
- **Multiple Loss Functions**: BCE, weighted BCE, and Focal Loss
- **Comprehensive Metrics**: Precision, recall, F1-score tracking with visualization

## Focal Loss: The Core Innovation

### The Problem with Standard Cross-Entropy

In fraud detection, the extreme class imbalance (99.8% legitimate vs 0.2% fraud) causes standard cross-entropy loss to:
- Focus excessively on the majority class
- Produce models that predict "not fraud" for everything
- Achieve high accuracy but miss actual fraud cases

### Focal Loss Solution

Focal Loss, introduced by Lin et al. in ["Focal Loss for Dense Object Detection"](https://arxiv.org/abs/1708.02002), addresses this by adding a modulating factor that down-weights easy examples:

```
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
```

Where:
- **p_t**: Probability of the true class
- **alpha (α)**: Class balancing weight (0-1). Higher values weight the minority class more heavily
- **gamma (γ)**: Focusing parameter (≥0). Higher values reduce the loss contribution from easy examples

### Implementation

From `src/focal_loss.py`:

```python
def compute_focal_loss(self, y_hat, y_true, alpha, gamma):
    # Step 1: Compute binary cross entropy (element-wise)
    bce_loss = F.binary_cross_entropy_with_logits(y_hat, y_true, reduction='none')

    # Step 2: Compute probability of true class
    # p_t = exp(-BCE) - high when prediction is correct, low when wrong
    p_t = torch.exp(-bce_loss)

    # Step 3: Compute focal weight
    # Easy examples (high p_t) get LOW weight
    # Hard examples (low p_t) get HIGH weight
    focal_weight = (1 - p_t) ** gamma

    # Step 4: Apply class-specific alpha weighting
    alpha_t = torch.where(y_true == 1, alpha, 1 - alpha)
    focal_weight = alpha_t * focal_weight

    # Step 5: Return mean focal loss
    return (focal_weight * bce_loss).mean()
```

### How It Works

| Example Type | p_t | (1-p_t)^γ | Effect |
|--------------|-----|-----------|--------|
| Easy (correct, confident) | 0.95 | 0.05^γ ≈ 0 | Nearly zero loss |
| Moderate | 0.70 | 0.30^γ | Reduced loss |
| Hard (incorrect/uncertain) | 0.30 | 0.70^γ | Full loss contribution |

With γ=2:
- A well-classified example (p_t=0.9) has its loss reduced by 100x
- A hard example (p_t=0.5) has its loss reduced by only 4x

## Project Structure

```
fraud-detection-neural-network/
├── README.md
├── src/
│   ├── __init__.py
│   ├── config.yaml          # All hyperparameters
│   ├── data_pipeline.py     # Data loading and preprocessing
│   ├── focal_loss.py        # Focal loss implementation
│   ├── model_pipeline.py    # PyTorch training pipeline
│   └── mlp.py               # MLP from scratch with NumPy
└── tests/
    ├── __init__.py
    └── test_mlp.py          # Unit tests
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fraud-detection-neural-network.git
cd fraud-detection-neural-network

# Install dependencies
pip install torch numpy pandas scikit-learn matplotlib pyyaml
```

## Usage

### Training with Focal Loss

```python
import yaml
from src.model_pipeline import ModelPipeline

# Load configuration
with open('src/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set to use focal loss
config['loss_fn'] = 'focalloss'
config['alpha'] = 0.6  # Class weight for positive class
config['gamma'] = 4.0  # Focusing parameter

# Initialize and train
pipeline = ModelPipeline(config, model_name="FraudDetector")
results = pipeline.train_and_evaluate(epochs=10)
```

### Hyperparameter Tuning

```python
from src.focal_loss import FocalLoss

# Initialize focal loss with config
focal = FocalLoss(config)

# Run randomized search over alpha and gamma
best_params = focal.focal_random_search(n_iterations=10)
```

### MLP from Scratch

```python
from src.mlp import MLPTwoLayers

# Create model
model = MLPTwoLayers(input_size=29, hidden_size=100, output_size=2)

# Training loop
for x, y in training_data:
    predictions = model.forward(x)
    loss = model.loss(predictions, y)
    model.backward()
```

## Model Architecture

The default neural network architecture:

```
Input (29 features)
    ↓
Linear(29 → 20) + ReLU + Dropout(0.3)
    ↓
Linear(20 → 12) + ReLU + Dropout(0.3)
    ↓
Linear(12 → 6) + ReLU + Dropout(0.3)
    ↓
Linear(6 → 3) + ReLU + Dropout(0.3)
    ↓
Linear(3 → 1)
    ↓
Output (logits → Sigmoid for probability)
```

Configurable options:
- **Architectures**: Dropout, BatchNorm, or plain
- **Activations**: ReLU, Sigmoid, Tanh, LeakyReLU, ELU
- **Initializers**: Xavier (uniform/normal), Kaiming (uniform/normal)
- **Optimizers**: Adam, SGD, RMSprop

## Configuration

Key parameters in `config.yaml`:

```yaml
# Focal Loss parameters
alpha: 0.6        # Higher = more weight to fraud class
gamma: 4.0        # Higher = more focus on hard examples

# Class weights (alternative to focal loss)
class_weights: [0.501, 289.4]  # ~1:578 ratio

# Model architecture
layer_sizes: [20, 12, 6, 3, 1]
dropout_rate: 0.3
```

## Results

With optimized focal loss (α=0.6, γ=4.0):
- Significantly improved recall for fraud detection
- Better F1-score compared to standard BCE loss
- Reduced false negatives (missed fraud cases)

## Technical Highlights

### Numerical Stability
- Log-sum-exp trick in softmax to prevent overflow
- Epsilon clipping in log operations to prevent -inf
- Gradient clipping in sigmoid to prevent vanishing gradients

### Loss Functions Compared

| Loss Function | Handles Imbalance | Focus on Hard Examples |
|---------------|-------------------|------------------------|
| BCE | No | No |
| Weighted BCE | Yes (static) | No |
| Focal Loss | Yes (adaptive) | Yes |

## References

- Lin, T.Y., et al. (2017). [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- He, K., et al. (2015). [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852) (He initialization)

## License

MIT License
