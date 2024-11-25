import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr  # learning rate
        self.activation_fn = activation  # activation function
        # Define layers and initialize weights
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initialize weights and biases
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2. / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2. / hidden_dim)
        self.b2 = np.zeros((1, output_dim))

        # For storing activations
        self.Z1 = None
        self.A1 = None
        self.Z2 = None
        self.A2 = None

        # For storing gradients
        self.dW1 = None
        self.db1 = None
        self.dW2 = None
        self.db2 = None

    def forward(self, X):
        # Forward pass: Input -> Hidden -> Output
        self.Z1 = X.dot(self.W1) + self.b1  # Linear step
        if self.activation_fn == 'tanh':
            self.A1 = np.tanh(self.Z1)  # Tanh activation
        elif self.activation_fn == 'relu':
            self.A1 = np.maximum(0, self.Z1)  # ReLU activation
        elif self.activation_fn == 'sigmoid':
            self.A1 = 1 / (1 + np.exp(-self.Z1))  # Sigmoid activation
        else:
            raise ValueError("Unsupported activation function")

        # Hidden layer to output layer
        self.Z2 = self.A1.dot(self.W2) + self.b2
        self.A2 = self.Z2 

        return self.A2 

    def backward(self, X, y):
        m = X.shape[0]

        # Derivative of loss with respect to output
        dA2 = (self.A2 - y) / m

        # Backpropagation into W2 and b2
        self.dW2 = self.A1.T.dot(dA2)
        self.db2 = np.sum(dA2, axis=0, keepdims=True)

        # Backpropagation into hidden layer
        dA1 = dA2.dot(self.W2.T)
        if self.activation_fn == 'tanh':
            dZ1 = dA1 * (1 - np.tanh(self.Z1) ** 2) 
        elif self.activation_fn == 'relu':
            dZ1 = dA1 * (self.Z1 > 0).astype(float)
        elif self.activation_fn == 'sigmoid':
            sigmoid_Z1 = 1 / (1 + np.exp(-self.Z1))
            dZ1 = dA1 * sigmoid_Z1 * (1 - sigmoid_Z1)
        else:
            raise ValueError("Unsupported activation function")

        # Backpropagation into W1 and b1
        self.dW1 = X.T.dot(dZ1)
        self.db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Update weights and biases using gradient descent
        self.W1 -= self.lr * self.dW1
        self.b1 -= self.lr * self.db1
        self.W2 -= self.lr * self.dW2
        self.b2 -= self.lr * self.db2

        # Store gradients for visualization
        self.grad_W1 = np.abs(self.dW1)
        self.grad_W2 = np.abs(self.dW2)


def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_input.clear()
    ax_hidden.clear()
    ax_gradient.clear()

    # Perform training steps by calling forward and backward function
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)

    # Hidden Space Visualization: Features, Distorted Input Space, and Decision Plane
    hidden_features = mlp.A1
    if mlp.hidden_dim >= 3:
        ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2],
                          c=(y.ravel() > 0), cmap='bwr', alpha=0.7)

        hs_min = hidden_features.min(axis=0)
        hs_max = hidden_features.max(axis=0)
        x_vals = np.linspace(hs_min[0] - 0.1, hs_max[0] + 0.1, 20)
        y_vals = np.linspace(hs_min[1] - 0.1, hs_max[1] + 0.1, 20)
        x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)

        W = mlp.W2[:, 0]
        if W[2] != 0:
            z_mesh = (-W[0] * x_mesh - W[1] * y_mesh - mlp.b2[0, 0]) / W[2]
            ax_hidden.plot_surface(x_mesh, y_mesh, z_mesh, alpha=0.3, color='orange')

        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xi, yi = np.meshgrid(np.linspace(x_min, x_max, 20), np.linspace(y_min, y_max, 20))
        grid = np.c_[xi.ravel(), yi.ravel()]

        # Transform grid points through the hidden layer
        Z1 = grid.dot(mlp.W1) + mlp.b1
        A1 = np.tanh(Z1) if mlp.activation_fn == 'tanh' else np.maximum(0, Z1)

        # Reshape for 3D plotting
        A1_x = A1[:, 0].reshape(xi.shape)
        A1_y = A1[:, 1].reshape(yi.shape)
        if mlp.hidden_dim >= 3:
            A1_z = A1[:, 2].reshape(xi.shape)
            ax_hidden.plot_surface(A1_x, A1_y, A1_z, color='purple', alpha=0.2)

        # Set axis limits to cover all points
        ax_hidden.set_xlim([hs_min[0] - 0.1, hs_max[0] + 0.1])
        ax_hidden.set_ylim([hs_min[1] - 0.1, hs_max[1] + 0.1])
        ax_hidden.set_zlim([hs_min[2] - 0.1, hs_max[2] + 0.1])

        ax_hidden.set_title(f"Hidden Space at Step {frame * 10}")
        ax_hidden.set_xlabel("h1")
        ax_hidden.set_ylabel("h2")
        ax_hidden.set_zlabel("h3")
    else:
        ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1],
                          c=(y.ravel() > 0), cmap='bwr', alpha=0.7)

        # Plot decision boundary in 2D
        hs_min = hidden_features.min(axis=0)
        hs_max = hidden_features.max(axis=0)
        x_vals = np.linspace(hs_min[0] - 0.1, hs_max[0] + 0.1, 100)
        W = mlp.W2[:, 0]
        if W[1] != 0:
            y_vals = (-W[0] * x_vals - mlp.b2[0, 0]) / W[1]
            ax_hidden.plot(x_vals, y_vals, 'g-', linewidth=2)

        # Set axis limits to cover all points
        ax_hidden.set_xlim([hs_min[0] - 0.1, hs_max[0] + 0.1])
        ax_hidden.set_ylim([hs_min[1] - 0.1, hs_max[1] + 0.1])

        ax_hidden.set_title(f"Hidden Space at Step {frame * 10}")
        ax_hidden.set_xlabel("h1")
        ax_hidden.set_ylabel("h2")

    # Input Space Visualization
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = mlp.forward(grid).reshape(xx.shape)

    ax_input.contourf(xx, yy, Z, levels=sorted([Z.min(), 0, Z.max()]), cmap='bwr', alpha=0.7)
    ax_input.scatter(X[:, 0], X[:, 1], c=(y.ravel() > 0), cmap='bwr', edgecolors='k')
    ax_input.set_title(f"Input Space at Step {frame * 10}")
    ax_input.set_xlabel("x1")
    ax_input.set_ylabel("x2")

    input_nodes = [(0, i) for i in range(mlp.input_dim)]
    hidden_nodes = [(1, i) for i in range(mlp.hidden_dim)]
    output_nodes = [(2, 0)]

    for x, y_pos in input_nodes:
        ax_gradient.add_patch(Circle((x, y_pos), 0.2, color='blue'))
    for x, y_pos in hidden_nodes:
        ax_gradient.add_patch(Circle((x, y_pos), 0.2, color='green'))
    for x, y_pos in output_nodes:
        ax_gradient.add_patch(Circle((x, y_pos), 0.2, color='red'))

    # Plot edges with gradient-based thickness
    for i in range(mlp.input_dim):
        for j in range(mlp.hidden_dim):
            grad = mlp.grad_W1[i, j]
            linewidth = grad * 150
            ax_gradient.plot([input_nodes[i][0], hidden_nodes[j][0]],
                             [input_nodes[i][1], hidden_nodes[j][1]], 'k-', linewidth=max(linewidth, 0.5), alpha=0.7)

    for i in range(mlp.hidden_dim):
        grad = mlp.grad_W2[i, 0]
        linewidth = grad * 150
        ax_gradient.plot([hidden_nodes[i][0], output_nodes[0][0]],
                         [hidden_nodes[i][1], output_nodes[0][1]], 'k-', linewidth=max(linewidth, 0.5), alpha=0.7)
    ax_gradient.set_title(f"Gradients at Step {frame * 10}")
    ax_gradient.axis('off')

def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    if mlp.hidden_dim >= 3:
        ax_hidden = fig.add_subplot(131, projection='3d')
    else:
        ax_hidden = fig.add_subplot(131)
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input,
                                     ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y),
                        frames=step_num // 10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "relu"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)
