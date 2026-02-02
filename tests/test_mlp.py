import logging
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mlp import MLPTwoLayers as MLP


def test_mlp_forward():
    """Test that forward pass produces valid probability distributions."""
    model = MLP(input_size=4, hidden_size=16, output_size=3)

    # Create random input
    x = np.random.randn(10, 4)

    # Forward pass
    output = model.forward(x)

    # Check output shape
    assert output.shape == (10, 3), f"Expected shape (10, 3), got {output.shape}"

    # Check that outputs are valid probabilities (sum to 1)
    sums = np.sum(output, axis=1)
    assert np.allclose(sums, 1.0), "Softmax outputs should sum to 1"

    # Check that all values are between 0 and 1
    assert np.all(output >= 0) and np.all(output <= 1), "Probabilities should be in [0, 1]"


def test_mlp_loss_decreases():
    """Test that loss decreases during training, indicating learning."""
    model = MLP(input_size=4, hidden_size=16, output_size=3)

    # Create simple training data
    np.random.seed(42)
    x = np.random.randn(150, 4)
    y = np.random.randint(0, 3, size=150)

    total_loss = 0
    all_losses = []

    for i in range(150):
        # Forward pass
        predictions = model.forward(x[i:i+1])
        loss = model.loss(predictions, y[i])
        all_losses.append(loss)
        total_loss += loss

        if i % 30 == 0:
            logging.info(f"Iteration {i}: Average loss {total_loss/(i+1):.4f}")

        # Backward pass (update weights)
        model.backward()

    # Check that loss decreased from start to end
    assert all_losses[0] > all_losses[-1], \
        f"Loss should decrease during training. Start: {all_losses[0]:.4f}, End: {all_losses[-1]:.4f}"


def test_mlp_weight_initialization():
    """Test that weights are properly initialized with He initialization."""
    model = MLP(input_size=100, hidden_size=50, output_size=10)

    # Check shapes
    assert model.w1.shape == (100, 50), f"W1 shape mismatch: {model.w1.shape}"
    assert model.b1.shape == (1, 50), f"b1 shape mismatch: {model.b1.shape}"
    assert model.w2.shape == (50, 10), f"W2 shape mismatch: {model.w2.shape}"
    assert model.b2.shape == (1, 10), f"b2 shape mismatch: {model.b2.shape}"

    # Check that biases are initialized to zero
    assert np.allclose(model.b1, 0), "b1 should be initialized to zeros"
    assert np.allclose(model.b2, 0), "b2 should be initialized to zeros"

    # Check that weights have reasonable variance (He initialization)
    expected_std_w1 = np.sqrt(2.0 / 100)
    actual_std_w1 = np.std(model.w1)
    assert 0.5 * expected_std_w1 < actual_std_w1 < 2 * expected_std_w1, \
        f"W1 std ({actual_std_w1:.4f}) should be close to He init std ({expected_std_w1:.4f})"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Running MLP tests...")
    test_mlp_forward()
    print("test_mlp_forward passed")

    test_mlp_loss_decreases()
    print("test_mlp_loss_decreases passed")

    test_mlp_weight_initialization()
    print("test_mlp_weight_initialization passed")

    print("\nAll tests passed!")
