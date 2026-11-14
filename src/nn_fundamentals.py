"""
Module: src.nn_fundamentals
Implements fundamental neural network concepts from scratch using NumPy:
- Perceptron & Multi-Layer Perceptron (MLP)
- Activation functions: ReLU, Sigmoid, Tanh and their derivatives
- Forward propagation, loss computation (Cross-Entropy, MSE)
- Backward propagation (analytical gradients)
- Optimizers: SGD and Momentum
"""

import numpy as np


class Activation:
    @staticmethod
    def sigmoid(x):
        clipped = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-clipped))

    @staticmethod
    def sigmoid_derivative(x):
        s = Activation.sigmoid(x)
        return s * (1.0 - s)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1.0 - np.tanh(x) ** 2

    @staticmethod
    def softmax(x):
        shift_x = x - np.max(x, axis=1, keepdims=True)
        exps = np.exp(shift_x)
        return exps / np.sum(exps, axis=1, keepdims=True)


class Loss:
    @staticmethod
    def cross_entropy(y_pred, y_true):
        """
        y_pred: (N, C) probabilities
        y_true: (N,) class indices
        """
        eps = 1e-12
        n_samples = y_pred.shape[0]
        p = np.clip(y_pred[np.arange(n_samples), y_true], eps, 1.0 - eps)
        return -np.mean(np.log(p))

    @staticmethod
    def cross_entropy_grad(y_pred, y_true):
        """
        Derivative of cross-entropy with respect to pre-softmax logits
        """
        n_samples = y_pred.shape[0]
        grad = y_pred.copy()
        grad[np.arange(n_samples), y_true] -= 1.0
        return grad / n_samples


class NumPyMLP:
    """
    2-Layer Multi-Layer Perceptron built purely from scratch using NumPy.
    Demonstrates forward propagation, loss calculation, backpropagation, and optimization.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, activation="relu", seed=42):
        np.random.seed(seed)
        self.activation_name = activation.lower()
        # He or Xavier initialization based on activation
        if self.activation_name == "relu":
            scale1 = np.sqrt(2.0 / input_dim)
            scale2 = np.sqrt(2.0 / hidden_dim)
        else:
            scale1 = np.sqrt(1.0 / input_dim)
            scale2 = np.sqrt(1.0 / hidden_dim)

        self.W1 = np.random.randn(input_dim, hidden_dim) * scale1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * scale2
        self.b2 = np.zeros((1, output_dim))

        # Velocity for momentum optimizer
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)

    def _activate(self, z):
        if self.activation_name == "relu":
            return Activation.relu(z)
        elif self.activation_name == "sigmoid":
            return Activation.sigmoid(z)
        elif self.activation_name == "tanh":
            return Activation.tanh(z)
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")

    def _activate_derivative(self, z):
        if self.activation_name == "relu":
            return Activation.relu_derivative(z)
        elif self.activation_name == "sigmoid":
            return Activation.sigmoid_derivative(z)
        elif self.activation_name == "tanh":
            return Activation.tanh_derivative(z)
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")

    def forward(self, X):
        self.X = X
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self._activate(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.probs = Activation.softmax(self.Z2)
        return self.probs

    def backward(self, y_true):
        # Gradient of loss w.r.t Z2 (logits)
        dZ2 = Loss.cross_entropy_grad(self.probs, y_true)

        # Gradients for layer 2
        self.dW2 = np.dot(self.A1.T, dZ2)
        self.db2 = np.sum(dZ2, axis=0, keepdims=True)

        # Backpropagation to hidden layer
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self._activate_derivative(self.Z1)

        # Gradients for layer 1
        self.dW1 = np.dot(self.X.T, dZ1)
        self.db1 = np.sum(dZ1, axis=0, keepdims=True)

    def step(self, lr=0.01, momentum=0.9, l2_reg=1e-4):
        # L2 Regularization (weight decay)
        self.dW2 += l2_reg * self.W2
        self.dW1 += l2_reg * self.W1

        # Momentum update
        self.v_W2 = momentum * self.v_W2 - lr * self.dW2
        self.v_b2 = momentum * self.v_b2 - lr * self.db2
        self.v_W1 = momentum * self.v_W1 - lr * self.dW1
        self.v_b1 = momentum * self.v_b1 - lr * self.db1

        self.W2 += self.v_W2
        self.b2 += self.v_b2
        self.W1 += self.v_W1
        self.b1 += self.v_b1

    def train_epoch(self, X, y, lr=0.01, momentum=0.9, l2_reg=1e-4):
        probs = self.forward(X)
        loss = Loss.cross_entropy(probs, y)
        self.backward(y)
        self.step(lr=lr, momentum=momentum, l2_reg=l2_reg)
        preds = np.argmax(probs, axis=1)
        acc = np.mean(preds == y)
        return loss, acc


def run_activation_experiment():
    """
    Simulates training on synthetic multi-class data across ReLU, Sigmoid, and Tanh
    to compare convergence and learning efficiency.
    """
    np.random.seed(42)
    n_samples, n_features, n_classes = 400, 32, 4
    X = np.random.randn(n_samples, n_features)
    # Generate non-linear ground truth
    y = (np.sum(X[:, :4] ** 2, axis=1) + np.sin(X[:, 4]) * 2).astype(int) % n_classes

    results = {}
    epochs = 60
    for act in ["relu", "sigmoid", "tanh"]:
        model = NumPyMLP(input_dim=n_features, hidden_dim=64, output_dim=n_classes, activation=act)
        history = []
        for ep in range(epochs):
            loss, acc = model.train_epoch(X, y, lr=0.05, momentum=0.9)
            history.append((loss, acc))
        final_loss, final_acc = history[-1]
        results[act] = {
            "final_loss": float(final_loss),
            "final_acc": float(final_acc),
            "history": history
        }
    return results


if __name__ == "__main__":
    print("Testing NumPy Neural Network Fundamentals...")
    results = run_activation_experiment()
    for act, res in results.items():
        print(f"Activation: {act.upper():<8} | Final Loss: {res['final_loss']:.4f} | Final Acc: {res['final_acc']*100:.1f}%")
