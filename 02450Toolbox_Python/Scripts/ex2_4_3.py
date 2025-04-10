# exercise 2.4.3
#%%
## Intro
"""
This is a small experiment where the exercise has a slightly different format than usual.
The purpose is to explore the best format of Python exercise in the course.

It is a long script. We suggest you run it usign the #%% feature 
in VScode which allows you to easily run parts at the time in interactive mode 
(similar to a Jupyter notebook yet still havign the full VScode/debugger available)

"""
import importlib_resources
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import zscore

#%%
## TASK A: Load the Wine dataset
filename = importlib_resources.files("dtuimldmtools").joinpath("data/wine.mat")

# Load data file and extract variables of interest
# Note the number of instances are: red wine (0) - 1599; white wine (1) - 4898. 
mat_data = loadmat(filename)
X = mat_data["X"]
y = mat_data["y"].squeeze()
C = mat_data["C"][0, 0]
M = mat_data["M"][0, 0]
N = mat_data["N"][0, 0]
attribute_names = [name[0][0] for name in mat_data["attributeNames"]]
attribute_names = [f"{a1}" for a1 in attribute_names[:]]
class_names = [cls[0][0] for cls in mat_data["classNames"]]
wine_id = np.arange(0, N)

#%%
## TASK B: Remove the outlies (as detected in a previous exercise)
if True: # try setting once you and see the effect on the distances
    outlier_mask = (X[:, 1] > 20) | (X[:, 7] > 10) | (X[:, 10] > 200)
    valid_mask = np.logical_not(outlier_mask)

    # Finally we will remove these from the data set
    X = X[valid_mask, :]
    y = y[valid_mask]
    wine_id = wine_id[valid_mask]
    N = len(y)

#%%
## TASK C: Randomly select row indices to make the analysis simpler
# You can change this if you want (the default is 100)
N_wines_to_consider = 100

np.random.seed(123) # we seed the random number generator to get the same random sample every time
subsample_mask = np.random.choice(N, N_wines_to_consider, replace=False)
X = X[subsample_mask, :]
y = y[subsample_mask]
wine_id = wine_id[subsample_mask] # this is simply so we can id the orginal winev if need be

sorted_indices = np.argsort(y) # sort rows in X acording to whether they are red of white
X = X[sorted_indices]
y = y[sorted_indices]
wine_id = wine_id[sorted_indices]
N = len(y)

# create a list of string for the plots xticks/labels
idx = np.arange(0,N)
wine_id_type = [f"{a3} (id={a1} type={a2})" for a1,a2,a3 in zip(wine_id, y , idx)]
wine_id_type_vert = [f"(id={a1} type={a2}) {a3}" for a1,a2,a3 in zip(wine_id, y , idx)]


#%%
## TASK D: Optionally, standardize the attributes
# Try, once you have completed the script, to change this and see the effect on
# the associated distance in TASK H and I
if True:
    X = zscore(X, ddof=1)

#%%
## TASK E: Show the attributes for insights
print("This is X:")
print(X)

fig = plt.figure(figsize=(10, 8))
plt.imshow(X, aspect='auto', cmap='jet')
plt.colorbar(label='Feature Values')
plt.title('Heatmap Data Matrix')
plt.yticks(ticks=np.arange(len(y)), labels=wine_id_type, fontsize=4)
plt.xticks(ticks=np.arange(len(attribute_names)), labels=attribute_names, rotation="vertical")
#plt.xticks(ticks=np.arange(len(attribute_names)), labels=wine_id_type, fontsize=4)
plt.xlabel('Attributes/features')
plt.ylabel('Observations')
plt.show()

print("Data loaded")

#%%
## TASK F: Extract two wines and compute distances between a white and red wine (warm up exercise)
#
# Experiment with the various scaling factors and attributes being scale 
# to see how the scaling affects the Lp distances (default L2)
#
# Note: you should think about ´x_red´ and ´x_white´ as vectors!
#
x_red = np.copy(X[0,:]) # note we make a copy to avoid messing with X in case we change x_white and x_red
x_white = np.copy(X[-1,:])
print("x_red: %s" % x_red)
print("x_white: %s" % x_white)
dist_firstandlast = np.linalg.norm(x_red - x_white, 2)  # L_2
print("Distance: %s  \n\n" % dist_firstandlast)

# Try to change the scale of one of the wines and see the effect on teh distance
sf = 1000
x_red = sf*np.copy(X[0,:])
x_white = sf*np.copy(X[-1,:])
print("x_red: %s" % x_red)
print("x_white: %s" % x_white)
dist_firstandlast = np.linalg.norm(x_red - x_white, 2)  # L_2
print(dist_firstandlast)
print("Distance after scaling all attributes: %s \n\n" % dist_firstandlast)

# Try to change the scale of one of the attributes in both wines and see the effect on the distance
x_red = np.copy(X[0,:])
x_white = np.copy(X[-1,:])
print("x_red: %s" % x_red)
print("x_white: %s" % x_white)
sf = 1000
x_white[1] = sf*x_white[1]
x_red[1] = sf*x_red[1]
print("x_red: %s" % x_red)
print("x_white: %s" % x_white)
dist_firstandlast = np.linalg.norm(x_red - x_white, 2)  # L_2
print("Distance after scaling one attribute: %s  \n\n" % dist_firstandlast)


#%% 
## TASK G: Compute and visualize distances between a wine and all others 
#
x_red = np.copy(X[0,:]) # note we make a copy to avoid messing with X in case we change x_white and x_red
x_white = np.copy(X[-1,:])

# we must use axis=1 to get the right result, otherwise the matrix norm will be used
# (the matrix norm is calculated across the whole matrix, rather than across each row vector!)
red_L1 = np.linalg.norm(X - x_red, 1, axis=1)  # L_1
red_L2 = np.linalg.norm(X - x_red, 2, axis=1)  # L_2
red_Linf = np.linalg.norm(X - x_red, np.inf, axis=1)  # L_inf

# This is not important 
def list_in_order(alist, order): # credit JHW
    """Given a list 'alist' and a list of indices 'order'
    returns the list in the order given by the indices. Credit: JHW"""
    return [alist[i] for i in order]

def rank_plot(distances):  # credit JHW
    """
    A helper function. Credit: JHW  
    """
    order = np.argsort(distances) # find the ordering of the distances    
    ax.bar(np.arange(len(distances)), distances[order]) # bar plot them
    ax.set_xlabel("Wines / type", fontsize=12)
    ax.set_ylabel("Distance to the first red wine", fontsize=12)
    ax.set_xticks(np.arange(N))
    #ax.set_frame_on(False) # remove frame
    # make sure the correct order is used for the labels!
    ax.set_xticklabels(
        list_in_order(wine_id_type, order), rotation="vertical", fontsize=7        
    )

# Make the plots (not important how this happens)
fig = plt.figure(figsize=(15, 22.5))
ax = fig.add_subplot(3, 1, 1)
ax.set_title("$L_2$ norm", fontsize=16)
rank_plot(red_L1)
ax = fig.add_subplot(3, 1, 2)
ax.set_title("$L_1$ norm", fontsize=16)
rank_plot(red_L2)
ax = fig.add_subplot(3, 1, 3)
ax.set_title("$L_\infty$ norm", fontsize=16)
rank_plot(red_Linf)
plt.tight_layout()



#%% 
## TASK H: Plot distances between all wines.
# Compute all the possible pairwise distances between rows and save 
# in the following variables:
#
# ´pairwise_distances_L1´: An NxN matrix with distances between row i and row j using L1
# ´pairwise_distances_L2´: An NxN matrix with distances between row i and row j using L2
# ´pairwise_distances_Linf´: An NxN matrix with distances between row i and row j using Linf
#

pairwise_distances_L1 = np.zeros((N, N))
pairwise_distances_L2 = np.zeros((N, N))
pairwise_distances_Linf = np.zeros((N, N))


# TASK: INSERT YOUR CODE HERE
raise NotImplementedError()


# Plot the pairwise distances as an image (not critical to understand the specific plotting code)
fig = plt.figure(figsize=(15, 22.5))
ax = fig.add_subplot(3, 1, 1)
cax=plt.imshow(pairwise_distances_L1, aspect='auto', cmap='jet')
plt.xticks(ticks=np.arange(len(y)), labels=wine_id_type_vert, fontsize=4, rotation="vertical")
plt.yticks(ticks=np.arange(len(y)), labels=wine_id_type, fontsize=4)
plt.title("Heatmap of Pairwise L1 Distances Between Observations")
plt.colorbar(cax, label="Distance")
ax.set_aspect('equal', 'box')

ax = fig.add_subplot(3, 1, 2)
cax=plt.imshow(pairwise_distances_L2, aspect='auto', cmap='jet')
plt.xticks(ticks=np.arange(len(y)), labels=wine_id_type_vert, fontsize=4, rotation="vertical")
plt.yticks(ticks=np.arange(len(y)), labels=wine_id_type, fontsize=4)
plt.title("Heatmap of Pairwise L2 Distances Between Observations")
plt.colorbar(cax, label="Distance")
ax.set_aspect('equal', 'box')

ax = fig.add_subplot(3, 1, 3)
cax=plt.imshow(pairwise_distances_Linf, aspect='auto', cmap='jet')
plt.xticks(ticks=np.arange(len(y)), labels=wine_id_type_vert, fontsize=4, rotation="vertical")
plt.yticks(ticks=np.arange(len(y)), labels=wine_id_type, fontsize=4)
plt.title("Heatmap of Pairwise Linf Distances Between Observations")
plt.colorbar(cax, label="Distance")
ax.set_aspect('equal', 'box')
plt.tight_layout()

plt.show()

#%%
## TASK I (i.e. i): Compute the following distances and store them in the approiate variables: 
#
# ´avg_interdist_white`: Average distance between all white wines based on the L1 norm (excluding distances to the same wine, i.e. 0)
# ´avg_interdist_red´: Average distance between all red wines based on the L1 norm (excluding distances to the same wine, i.e. 0)
# ´avg_intradist_red2white´: Average distance between white and red and white wines based on the L1 norm
# 
# Hint: You can obtain the required information from the ´pairwise_distances´ variables
# above
#
# Question: Describe how the informaton about average inter and intra distances 
# can be used in (automatically) disciminating between white and red wines?
#
# Question: Does it make a difference if you use the L1, L2 or Linf norm? Consider the
# relative difference between the inter and intra wine distances (p.s. it does...). 
#

avg_interdist_white = np.nan # replace np.nan with your estimate
avg_interdist_red = np.nan # replace np.nan with your estimate
avg_intradist_red2white = np.nan # replace np.nan with your estimate


# TASK: INSERT YOUR CODE HERE
raise NotImplementedError()


#%%
print("You are now done with this exercise. Ask your TA to look over your solutions and discuss your findings with them.")#%%
# %%
