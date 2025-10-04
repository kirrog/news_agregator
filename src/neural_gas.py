import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

class NeuralGas:
    """
    Neural Gas algorithm for clustering
    """

    def __init__(self, n_units, max_iter=100, lambda_i=10, lambda_f=0.01,
                 epsilon_i=0.5, epsilon_f=0.05, random_state=None):
        """
        Initialize Neural Gas

        Parameters:
        - n_units: Number of prototype vectors (clusters)
        - max_iter: Maximum number of iterations
        - lambda_i, lambda_f: Initial and final values for neighborhood range
        - epsilon_i, epsilon_f: Initial and final learning rates
        - random_state: Random seed for reproducibility
        """
        self.n_units = n_units
        self.max_iter = max_iter
        self.lambda_i = lambda_i
        self.lambda_f = lambda_f
        self.epsilon_i = epsilon_i
        self.epsilon_f = epsilon_f
        self.random_state = random_state
        self.prototypes_ = None
        self.labels_ = None

    def _initialize_prototypes(self, X):
        """Initialize prototype vectors randomly from the data"""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Randomly select n_units data points as initial prototypes
        indices = np.random.choice(X.shape[0], self.n_units, replace=False)
        self.prototypes_ = X[indices].copy()

    def _calculate_neighborhood_rankings(self, x):
        """Calculate neighborhood rankings for a given data point"""
        # Calculate distances to all prototypes
        distances = np.linalg.norm(self.prototypes_ - x, axis=1)

        # Get indices sorted by distance (0 = closest, n_units-1 = farthest)
        rankings = np.argsort(distances)

        return rankings

    def _update_prototypes(self, x, iteration):
        """Update prototype vectors for a given data point"""
        # Calculate current learning rate and neighborhood range
        t = iteration / self.max_iter
        epsilon_t = self.epsilon_i * (self.epsilon_f / self.epsilon_i) ** t
        lambda_t = self.lambda_i * (self.lambda_f / self.lambda_i) ** t

        # Get neighborhood rankings
        rankings = self._calculate_neighborhood_rankings(x)

        # Update each prototype
        for k, rank in enumerate(rankings):
            # Neighborhood function: h(k) = exp(-k / lambda_t)
            h = np.exp(-rank / lambda_t)

            # Update prototype
            self.prototypes_[k] += epsilon_t * h * (x - self.prototypes_[k])

    def fit(self, X):
        """
        Fit the Neural Gas model to the data

        Parameters:
        - X: Input data of shape (n_samples, n_features)
        """
        # Initialize prototypes
        self._initialize_prototypes(X)

        # Training loop
        for iteration in range(self.max_iter):
            # Shuffle data for each epoch
            indices = np.random.permutation(X.shape[0])

            for idx in indices:
                x = X[idx]
                self._update_prototypes(x, iteration)

            # Optional: Print progress
            if (iteration + 1) % 20 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iter}")

        # Assign labels based on closest prototype
        self.labels_ = self.predict(X)

        return self

    def predict(self, X):
        """
        Predict cluster labels for input data

        Parameters:
        - X: Input data of shape (n_samples, n_features)

        Returns:
        - labels: Cluster labels for each data point
        """
        if self.prototypes_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        labels = []
        for x in X:
            distances = np.linalg.norm(self.prototypes_ - x, axis=1)
            labels.append(np.argmin(distances))

        return np.array(labels)

    def quantization_error(self, X):
        """
        Calculate quantization error

        Parameters:
        - X: Input data

        Returns:
        - error: Average distance to closest prototype
        """
        labels = self.predict(X)
        error = 0.0

        for i, x in enumerate(X):
            closest_prototype = self.prototypes_[labels[i]]
            error += np.linalg.norm(x - closest_prototype)

        return error / len(X)


# Example usage and visualization
def demonstrate_neural_gas():
    """Demonstrate the Neural Gas algorithm on synthetic data"""

    # Generate sample data
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60,
                           random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply Neural Gas
    ng = NeuralGas(n_units=4, max_iter=100, lambda_i=10, lambda_f=0.01,
                   epsilon_i=0.5, epsilon_f=0.05, random_state=42)
    ng.fit(X_scaled)

    # Predict clusters
    y_pred = ng.predict(X_scaled)

    # Calculate quantization error
    error = ng.quantization_error(X_scaled)
    print(f"Quantization error: {error:.4f}")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # True clusters
    ax1.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.7)
    ax1.set_title('True Clusters')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')

    # Neural Gas clusters
    ax2.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
    ax2.scatter(scaler.inverse_transform(ng.prototypes_)[:, 0],
                scaler.inverse_transform(ng.prototypes_)[:, 1],
                c='red', marker='X', s=200, label='Prototypes')
    ax2.set_title('Neural Gas Clustering')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return ng, X, y_true, y_pred


# Additional utility function for finding optimal number of clusters
def find_optimal_clusters(X, max_clusters=10):
    """
    Find optimal number of clusters using elbow method with quantization error
    """
    errors = []
    cluster_range = range(2, max_clusters + 1)

    for n_clusters in cluster_range:
        ng = NeuralGas(n_units=n_clusters, max_iter=50, random_state=42)
        ng.fit(X)
        error = ng.quantization_error(X)
        errors.append(error)
        print(f"Clusters: {n_clusters}, Error: {error:.4f}")

    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_range, errors, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Quantization Error')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.grid(True)
    plt.show()

    return errors


if __name__ == "__main__":
    # Demonstrate the algorithm
    ng_model, X_data, true_labels, pred_labels = demonstrate_neural_gas()

    # Optional: Find optimal number of clusters
    # Standardize data first


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data)
    find_optimal_clusters(X_scaled, max_clusters=24)
