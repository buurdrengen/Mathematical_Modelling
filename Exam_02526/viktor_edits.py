import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import skimage.measure
import paralleltomo # [A,theta,p,d] = paralleltomo(N,theta,p,d)



# Define the perbuations
noise = 1e-3
angle_no = 180
p = 200
res = 100 # The picture will be (res x res)


# Load the image with lead and steel shot
data = np.array(scipy.io.loadmat("Exam_02526/testImage.mat")['im']) #Pixel size: 0.1 mm, 5000x5000 pixels
data[data == np.unique(data)[[1]]] = 0.1*1e-2 # Change the tree attenuation
data[data == np.unique(data)[[2]]] = 1.14*1e-2 # Change the steel (iron) attenuation
data[data == np.unique(data)[[3]]] = 11.3*1e-2 # Change the lead attenuation

# Downsample image
downsizing = int(5e3/res) # define downsizing constant
downsized_im = skimage.measure.block_reduce(data, block_size=(downsizing, downsizing), func=np.mean) # downsample the image
x = np.copy(downsized_im.flatten()) # flatten the attenuation constants
print(f"the resolution of the image is {np.shape(downsized_im)}")

# Initialize system matrix and stuff
theta = np.array([np.linspace(0, 180, angle_no)])
d = 0.5 # [m]
N = downsized_im.shape[0]
[A,theta,p,d] = paralleltomo.paralleltomo(N, theta, p)


# Load simulated forward projection 
b = A @ x # Finding b
b_perturbed = b + np.random.normal(0, noise, size=np.shape(b)) # adding noise

# Find the perturbed attenuation coefficients
x_new = np.linalg.solve(A.T @ A, A.T @ b_perturbed)
x_new = x_new.reshape(np.shape(downsized_im))

# Define the attenuation coefficient means and find the separators betweeen classes
[air_mean, tree_mean, iron_mean, lead_mean] = np.unique(data)
print([air_mean, tree_mean, iron_mean, lead_mean])

# Find the point where the classes have equal probability of being
air_tree_separator = (air_mean + tree_mean)/2
tree_iron_separator = (tree_mean + iron_mean)/2
iron_lead_separator = (iron_mean + lead_mean)/2

# Find the index for the different classes
air_index =                                 (x_new < air_tree_separator)
tree_index = (x_new > air_tree_separator) & (x_new < tree_iron_separator)
iron_index = (x_new > tree_iron_separator) & (x_new < iron_lead_separator)
lead_index = (x_new > iron_lead_separator)


# Define a class tree to store the different classes identified
class_tree = np.zeros(np.shape(x_new))
class_tree[air_index] = 1
class_tree[tree_index] = 2
class_tree[iron_index] = 3
class_tree[lead_index] = 4

# Create the figure to show the different classes of materials
fig, ax = plt.subplots()
cmap = plt.get_cmap('viridis', 4) # Define the colormap
im = ax.imshow(class_tree, cmap=cmap, vmin=0.5, vmax=4.5) # Plot the image
cbar = fig.colorbar(im, ticks=[1, 2, 3, 4]) # Define colorbar
cbar.ax.set_yticklabels(['Air', 'Tree', 'Iron', 'Lead']) # Define tick labels
ax.set_title(f'No. of rays: {p}\nNo. of angles: {angle_no}')



plt.show()










