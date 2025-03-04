import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import svd 
import itertools

# Load dataset
seeds_dataset = "data/seeds_dataset.txt"
data = np.loadtxt(seeds_dataset)

# Extract features and labels
X = data[:, :-1]  # Features (all columns except last)
y = data[:, -1].astype(int)  # Class labels (last column, converted to int)

# Attribute and class names
attributeNames = ["area_A", "perimeter_P", "compactness_C", "length_of_kernel",
                  "width_of_kernel", "asymmetry_coefficient", "length_of_kernel_groove"]
classNames = ["Kama", "Rosa", "Canadian"]
colors = ["r", "g", "b"]
num_attributes = len(attributeNames)

# Create a figure with subplots
fig, axes = plt.subplots(num_attributes, num_attributes, figsize=(10, 10))
plt.suptitle("Pairwise Attribute Scatter Plots", fontsize=10)

# Loop through all attribute pairs
for i, j in itertools.product(range(num_attributes), repeat=2):
    ax = axes[i, j]
    
    # If same attribute, write its name in the middle
    if i == j:
        ax.text(0.5, 0.5, attributeNames[i], fontsize=10, ha='center', va='center')
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        # Plot scatter points for each class
        for c, color in zip(range(1, 4), colors):  # Class labels are 1,2,3
            mask = (y == c)  # Boolean mask for class c
            ax.scatter(X[mask, j], X[mask, i], label=classNames, color=color, alpha=0.6, s=5)
    
    # Set axis labels only on the edges
    if j == 0:
        ax.set_ylabel(attributeNames[i])
    if i == num_attributes - 1:
        ax.set_xlabel(attributeNames[j])

fig.legend(classNames)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()



# Create a new figure for the 3D scatter plot
fig1 = plt.figure(figsize=(10, 7))
ax1 = fig1.add_subplot(111, projection='3d')

# Choose three attributes (modify indices as needed)
a, b, d = 6, 1, 2  # Example: First three attributes

# Plot each class separately
for c1, color1 in zip(range(1, 4), colors):  # Classes are labeled 1, 2, 3
    mask = (y == c1)
    ax1.scatter(X[mask, a], X[mask, b], X[mask, d], 
                                      color=color1, alpha=0.6, s=20)

# Labels and title
ax1.set_xlabel(attributeNames[a])
ax1.set_ylabel(attributeNames[b])
ax1.set_zlabel(attributeNames[d])
ax1.set_title("3D Scatter Plot of Seed Attributes")
ax1.legend(classNames)
plt.show()



# Standardize the data
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_standardized = (X - X_mean) / X_std

# Perform SVD
U, S, Vt = svd(X_standardized, full_matrices=False)

# Principal directions
principal_components = Vt.T

# Plot principal components
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(principal_components.shape[1]):
    ax.plot(principal_components[:, i], label=f"PC{i+1}")
ax.set_title("Principal Directions")
ax.set_xlabel("Attributes")
ax.set_ylabel("Principal Component Coefficients")
ax.legend()
plt.xticks(range(num_attributes), attributeNames, rotation=45)
plt.grid()
plt.show()

# Explained variance
explained_variance = (S**2) / np.sum(S**2)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), marker='o', linestyle='--')
ax.set_title("Explained Variance by Number of Principal Components")
ax.set_xlabel("Number of Components")
ax.set_ylabel("Cumulative Explained Variance")
ax.grid()
plt.show()

# Project data onto first two principal components
X_pca = X_standardized @ principal_components[:, :2]

# Plot 2D projection
fig, ax = plt.subplots(figsize=(10, 7))
for c, color in zip(range(1, 4), colors):
    mask = (y == c)
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=classNames[c-1], color=color, alpha=0.6, s=20)
ax.set_title("2D Projection onto First Two Principal Components")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend()
plt.grid()
plt.show()

print("Script End")








