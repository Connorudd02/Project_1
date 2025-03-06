# %%
import numpy as np
import matplotlib.pyplot as plt
import dtuimldmtools as dtu
from scipy import stats
import itertools

# %% [markdown]
# # Seeds project

# %%
data_path = "data/"
seeds_dataset = "seeds_dataset.txt"
dataset_file = data_path + seeds_dataset

# %% [markdown]
# ### Import data

# %%
data = np.loadtxt(dataset_file)
# Validate shape of the dataset, 210 rows with 8 attributes
data.shape

# %% [markdown]
# ### Convert dataset based on course conventions
# ![Course conventions](images/course_conventions.png)

# %% [markdown]
# #### Data Fields
# Dataset obtained from https://archive.ics.uci.edu/dataset/236/seeds 
# * area_A: tensor containing an area of the wheat grains
# * perimeter_P: tensor containing the perimeter of the wheat grains
# * compactness_C: tensor containing compactness of the wheat grains
# * length_of_kernel: tensor containing the length of each wheat kernel
# * width_of_kernel: tensor containing the width of each wheat kernel
# * asymmetry_coefficient: tensor containing asymmetry coefficient of a wheat kernel
# * length_of_kernel_groove: tensor containing a length of a kernel groove

# %%
X = data
# attributeNames are not present in the dataset, just gonna hardcode based on the website
attributeNames = ["area_A", "perimeter_P", "compactness_C", "length_of_kernel", "width_of_kernel", "asymmetry_coefficient", "length_of_kernel_groove", "class"]
N = data.shape[0]
M = data.shape[1]
y = X[:, -1]
# This is derived from the website
classNames = ["Kama", "Rosa", "Canadian"]
C = len(classNames)
attributeNames, N, M, y, y.shape, classNames, C

# %% [markdown]
# ## Edit class labels, ensure zero indexing 

# %%
X[:, -1] -= 1
X.shape, X[:, -1]

# %% [markdown]
# ## Summary statistics

# %%
# Get mean, standard deviation and range for all attributes(Except labels)
for i in range(M-1):
    mean = np.mean(X[:, i])
    std = np.std(X[:, i])
    attribute_range = np.ptp(X[:, i])
    print(f"{attributeNames[i]}: mean is {mean}, std is {std} and range is {attribute_range}")


# %% [markdown]
# ## Basic plots

# %% [markdown]
# ### Plot attribute pairs

# %%
num_attributes = len(attributeNames)
colors = ["r", "g", "b"]
fig, axes = plt.subplots(num_attributes, num_attributes, figsize=(16, 16))
plt.suptitle("Pairwise Attribute Scatter Plots", fontsize=10)

# Loop through all attribute pairs
for i, j in itertools.product(range(num_attributes), repeat=2):
    ax = axes[i, j]

    # If same attribute, write its name in the middle
    if i == j:
        ax.text(
            0.5,
            0.5,
            attributeNames[i],
            fontsize=10,  # TODO Based on our use case, y.shape can be either (210, ) or (210, 1)
            ha="center",
            va="center",
        )
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        # Plot scatter points for each class
        for c, color in zip(range(3), colors):  # Class labels are 1,2,3
            mask = y == c  # Boolean mask for class c
            ax.scatter(
                X[mask, j], X[mask, i], label=classNames, color=color, alpha=0.6, s=5
            )

    # Set axis labels only on the edges
    if j == 0:
        ax.set_ylabel(attributeNames[i])
    if i == num_attributes - 1:
        ax.set_xlabel(attributeNames[j])

fig.legend(classNames)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("images/pairwiseplot.png")
plt.show()

# %% [markdown]
# ### Calculate correlation

# %%
import pandas as pd

# %%
df = pd.DataFrame(X, columns=attributeNames)
correlation_matrix = df.corr().values
plt.figure(figsize=(10, 8))
plt.title("Correlation Heatmap", fontsize=12, fontweight="bold")
heatmap = plt.imshow(
    correlation_matrix, cmap="coolwarm", interpolation="nearest", vmin=-1, vmax=1
)
plt.colorbar(heatmap, label="Correlation Coefficient")
plt.xticks(
    np.arange(len(attributeNames)), attributeNames, rotation=45, ha="right", fontsize=10
)
plt.yticks(np.arange(len(attributeNames)), attributeNames, fontsize=10)
for i in range(len(attributeNames)):
    for j in range(len(attributeNames)):
        plt.text(
            j,
            i,
            f"{correlation_matrix[i, j]:.3f}",
            ha="center",
            va="center",
            color="black",
            fontsize=8,
        )

plt.tight_layout()
plt.savefig("images/correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 3D scatter plot

# %%
# Create a new figure for the 3D scatter plot
fig1 = plt.figure(figsize=(10, 7))
ax1 = fig1.add_subplot(111, projection="3d")

# Choose three attributes (modify indices as needed)
a, b, d = 6, 1, 2  # Example: First three attributes

# Plot each class separately
for c1, color1 in zip(range(3), colors):  # Classes are labeled 1, 2, 3
    mask = y == c1
    ax1.scatter(X[mask, a], X[mask, b], X[mask, d], color=color1, alpha=0.6, s=20)

# Labels and title
ax1.set_xlabel(attributeNames[a])
ax1.set_ylabel(attributeNames[b])
ax1.set_zlabel(attributeNames[d])
ax1.set_title("3D Scatter Plot of Seed Attributes")
ax1.legend(classNames)
plt.savefig("images/3dscatter.png")
plt.show()

# %% [markdown]
# ### Histogram

# %%
plt.figure(figsize=(16, 14))
# Exclude the last data point because not necessary to plot classes in histogram
numData = M - 1 
u = np.floor(np.sqrt(numData))
v = np.ceil(float(numData) / u)
for i in range(numData):
    plt.subplot(int(u), int(v), i + 1)
    plt.hist(X[:, i], color=(0.2, 0.8 - i * 0.1, 0.4))
    plt.xlabel(attributeNames[i])
    plt.ylim(0, N / 2)

plt.savefig("images/histogram.png")
plt.show()

# %% [markdown]
# ## Remove outlier from the seed with extremely low kernel length

# %%
attribute_index = attributeNames.index("length_of_kernel")
lowest_index = np.argmin(X[:, 3])
X_updated = np.delete(X, lowest_index, axis=0)
y = np.delete(y, lowest_index, axis=0)
N -= 1
N, X_updated.shape, y.shape

# %% [markdown]
# ### Corrected histogram

# %%
plt.figure(figsize=(16, 14))
# Exclude the last data point because not necessary to plot classes in histogram
numData = M - 1
u = np.floor(np.sqrt(numData))
v = np.ceil(float(numData) / u)
for i in range(numData):
    plt.subplot(int(u), int(v), i + 1)
    plt.hist(X_updated[:, i], color=(0.2, 0.8 - i * 0.1, 0.4))
    plt.xlabel(attributeNames[i])
    plt.ylim(0, N / 2)

plt.savefig("images/histogram.png")
plt.show()

# %% [markdown]
# ### Standardize data

# %%
# Standardize the data
X_mean = np.mean(X_updated, axis=0)
X_std = np.std(X_updated, axis=0)
X_standardized = (X_updated - X_mean) / X_std

# %%
plt.figure(figsize=(16, 10))
plt.boxplot(X_standardized[:, :-1])
plt.xticks(range(1, numData+1), attributeNames[:-1])
plt.title("Seeds data set - boxplot")
plt.savefig("images/boxplot.png")
plt.show()

# %% [markdown]
# ### Box plot

# %%
plt.figure(figsize=(14, 7))
for c in range(C):
    plt.subplot(1, C, c + 1)
    class_mask = y == c
    plt.boxplot(X_standardized[class_mask, :])
    plt.title("Class: " + classNames[c])
    plt.xticks(range(1, numData + 1), attributeNames[:-1], rotation=45)
    y_up = X_standardized.max() + (X_standardized.max() - X_standardized.min()) * 0.1
    y_down = X_standardized.min() - (X_standardized.max() - X_standardized.min()) * 0.1
    plt.ylim(y_down, y_up)

plt.savefig("images/boxplt_classes.png")
plt.show()

# %% [markdown]
# ## PCA on Seeds dataset

# %%
# Remove the labels from the dataset because they should not be factored into PCA analysis
X_no_label = X_standardized[:, :-1]
X_no_label.shape

# %%
# Subtract

Y = X_no_label - np.ones((N, 1)) * X_no_label.mean(axis=0)
# SVD
U, S, Vh = np.linalg.svd(Y, full_matrices=False)
principal_components = Vh.T
principal_components

# %% [markdown]
# ### Plot of all principal directions

# %%
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
plt.savefig("images/principaldirections.png")
plt.show()

# %% [markdown]
# ### A plot of the amount of variance explained as a function of the number of PCA components included

# %%
rho = (S * S) / (S * S).sum()

threshold = 0.90

# Plot variance explained
plt.figure()
plt.plot(range(1, len(rho) + 1), rho, "x-")
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-")
plt.plot([1, len(rho)], [threshold, threshold], "k--")
plt.title("Variance explained by principal components")
plt.xlabel("Principal component")
plt.ylabel("Variance explained")
plt.legend(["Individual", "Cumulative", "Threshold"])
plt.grid()
plt.savefig("images/variancetallied.png")
plt.show()

# %% [markdown]
# First three principal components explains more than 90% variance of the data

# %% [markdown]
# ### Table of component coefficients

# %%
pcs = [0, 1, 2]
legendStrs = ["PC" + str(e + 1) for e in pcs]
box_width = 0.2
r = np.arange(0, M - 1)
print(r)
for i in pcs:
    plt.bar(r + i * box_width, principal_components[:, i], width=box_width)

plt.xticks(r, attributeNames[:-1], rotation=70)
plt.xlabel("Attributes")
plt.ylabel("Component coefficients")
plt.legend(legendStrs)
plt.grid()
plt.title("Seeds: PCA Component Coefficients")
plt.savefig("images/PCAcoefficients.png")
plt.show()

# %% [markdown]
# ### Projection of first two components

# %%
X_standardized[:, :-1].shape, principal_components.shape

# %%
# Project data onto first two principal components
X_pca = X_standardized[:, :-1] @ principal_components[:, :2]

# Plot 2D projection
fig, ax = plt.subplots(figsize=(10, 7))
for c, color in zip(range(3), colors):
    mask = y == c
    ax.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label=classNames[c],
        color=color,
        alpha=0.6,
        s=20,
    )
ax.set_title("2D Projection of First Two Principal Components")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend()
plt.grid()
plt.savefig("images/pca2d.png")
plt.show()

# %% [markdown]
# ### The data projected onto the first three principal components

# %%
# Project the centered data onto the principle component space
Z = np.dot(Y, principal_components)
# Add back class labels for dataset
figure = plt.figure(figsize=(10, 8))
ax = figure.add_subplot(projection='3d')
plt.title("Seeds dataset: PCA of first three components")
for c in range(C):
    class_mask = y == c
    ax.scatter(Z[class_mask, 0], Z[class_mask, 1], Z[class_mask, 2], "o", alpha=0.7)
ax.legend(classNames)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.tight_layout()
plt.savefig("images/pca3d.png")
plt.show()

# %% [markdown]
# Before standardization: The attribute with the highest standard deviation will dominate the PCA results because PCA is sensitive to the scale of the data. This attribute will likely have a large magnitude in the attribute coefficients (loadings) of the principal components.
# 
# After standardization: All attributes will have the same scale (mean = 0, standard deviation = 1). The attribute with the highest standard deviation will no longer dominate, and its contribution to the principal components will be more balanced with the other attributes.
# 
# Changes in direction and magnitude:
# 
# The direction of the attribute coefficients may change because the relative importance of the attributes is reweighted.
# 
# The magnitude of the coefficients for the previously dominant attribute will decrease after standardization.

# %% [markdown]
# Before standardization: The principal components will likely explain a large proportion of the variance in the dominant attribute, but this may not reflect the true structure of the data.
# 
# After standardization: The variance explained by the principal components will be more evenly distributed across all attributes, providing a more balanced representation of the data.


