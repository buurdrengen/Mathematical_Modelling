from pickle import NONE
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.linalg import solve
from scipy.stats import alpha
from skimage.measure import block_reduce
from paralleltomo import * # [A,theta,p,d] = paralleltomo(N,theta,p,d)
import time


# Define the perbuations
# noise = 8e-5
angle_no = 90
p = 120
res = 50 # The picture will be (res x res)
sample_size = 20
confidence = 2



# Load the image with lead and steel shot
data = np.array(loadmat("Exam_02526/testImage.mat")['im']) #Pixel size: 0.1 mm, 5000x5000 pixels
# data[data == np.unique(data)[[1]]] = 0.1*1e-2 # Change the tree attenuation
# data[data == np.unique(data)[[2]]] = 1.14*1e-2 # Change the steel (iron) attenuation
# data[data == np.unique(data)[[3]]] = 11.3*1e-2 # Change the lead attenuation

base_array = np.zeros(np.shape(data))
known_wood = np.copy(base_array); known_wood[data == np.unique(data)[[1]]] = 1
known_iron= np.copy(base_array); known_iron[data == np.unique(data)[[2]]] = 1
known_lead = np.copy(base_array); known_lead[data == np.unique(data)[[3]]] = 1

# Downsample image
downsizing = int(5e3/res) # define downsizing constant
downsized_im = block_reduce(data, block_size=(downsizing, downsizing), func=np.mean) # downsample the image
x = np.copy(downsized_im.flatten()) # flatten the attenuation constants

downsized_known_wood = block_reduce(known_wood, block_size=(downsizing, downsizing), func=np.mean) # downsample the image
downsized_known_iron = block_reduce(known_iron, block_size=(downsizing, downsizing), func=np.mean) # downsample the image
downsized_known_lead = block_reduce(known_lead, block_size=(downsizing, downsizing), func=np.mean) # downsample the image

downsized_known_lead = np.ceil(downsized_known_lead).astype(int) # Any cell containing lead will be treated as lead
downsized_known_iron = np.ceil(downsized_known_iron).astype(int) # Any cell containing iron will be treated as iron
downsized_known_wood = np.floor(downsized_known_wood + 0.01).astype(int) # Any cell containing at least 99% wood will be treated as wood

# Find the separators based on the anotted masks
x_imShape = x.reshape(np.shape(downsized_im)) 
# Define the attenuation coefficient means and find the separators betweeen classes
air_mean = 0
tree_mean = np.mean(x_imShape[downsized_known_wood == 1])
iron_mean = np.mean(x_imShape[downsized_known_iron == 1])
lead_mean = np.mean(x_imShape[downsized_known_lead == 1]) 
# Find the point where the classes have equal probability of being
air_tree_separator = (air_mean + tree_mean)/2
tree_iron_separator = (tree_mean + iron_mean)/2
iron_lead_separator = (iron_mean + lead_mean)/2

# number of each cell
N_lead = np.sum(downsized_known_lead)
N_iron = np.sum(downsized_known_iron)
N_wood = np.sum(downsized_known_wood)

print(f"the resolution of the image is {np.shape(downsized_im)}")

# Initialize system matrix and stuff
theta = np.array([np.linspace(0, 180, angle_no)]) 
d = 0.5 # [m]
N = downsized_im.shape[0]
[A,theta,p,d] = paralleltomo(N, theta, p)


# Load simulated forward projection 
b = A @ x # Finding bj
t0 = time.time()
error_list = np.arange(1,41)*25e-6
wood_errors = np.zeros([np.size(error_list),sample_size])
failed_to_detect_wood = -1
failed_to_detect_metal = -1
for i, noise in enumerate(error_list):
    for j in range(sample_size):
        b_perturbed = b + np.random.normal(0, noise, size=np.shape(b)) # adding noise

        # Find the perturbed attenuation coefficients
        x_new = solve(A.T @ A, A.T @ b_perturbed, assume_a = "her")
        x_new = x_new.reshape(np.shape(downsized_im)) 

        # Find the index for the different classes
        air_index =                                 (x_new < air_tree_separator)
        tree_index = (x_new > air_tree_separator) & (x_new < tree_iron_separator)
        iron_index = (x_new > tree_iron_separator) & (x_new < iron_lead_separator)
        lead_index = (x_new > iron_lead_separator)

        lead_error = np.sum(lead_index[downsized_known_lead == 1] == 0)/N_lead
        iron_error = np.sum(iron_index[downsized_known_iron == 1] == 0)/N_iron
        wood_error = np.sum(tree_index[downsized_known_wood == 1] == 0)/N_wood
        wood_errors[i,j] = (wood_error*N_wood + iron_error*N_iron + lead_error*N_lead)/(N_lead + N_iron + N_wood)
        
        if (wood_error > 0) & (failed_to_detect_wood == -1): failed_to_detect_wood = noise; print(f"FAIL: (at {i}) Failed To Detect Wood...")
        if (lead_error + iron_error > 0) & (failed_to_detect_metal == -1): failed_to_detect_metal = noise; print(f"FAIL: (at {i}) Failed To Detect Metal...")

    print(f"Error[{i}] is {np.mean(wood_errors[i,:])} pm {confidence*np.std(wood_errors[i,:])}")

t1 = time.time()
print(f"Time is {t1 - t0} seconds")

mean_error = np.mean(wood_errors,axis=1)
std_error = np.std(wood_errors,axis=1)

plt.fill_between(error_list, y1 = (mean_error + confidence*std_error)*100, y2 = (mean_error - confidence*std_error)*100, color="gray", lw=0, alpha=0.3, label="Confidence Interval")
plt.plot(error_list, mean_error*100, label="Error Rate", lw=2)
if failed_to_detect_wood >= 0: plt.axvline(failed_to_detect_wood, ls="--", color="orange", label="First False Positive")
if failed_to_detect_metal >= 0: plt.axvline(failed_to_detect_metal, ls="--", color="red", label="First False Negative")
plt.ylabel("Error Rate [%]", fontsize=14)
plt.xlabel("Added noise level", fontsize=14)
plt.ticklabel_format(axis='x', style='sci', scilimits=(-4,-4), useMathText=True)
plt.xlim([0,1e-3])
plt.ylim([0,50])
plt.tick_params(labelsize=11)
plt.legend(fontsize=12)
plt.grid()
plt.title(f'Resolution: {res}X{res}\nSetup: {p} Rays, {angle_no} Angles and {sample_size} Samples', fontsize=16)
plt.savefig(f"Exam_02526/img/res{res}.png", dpi=300)
plt.show()

# # Define a class tree to store the different classes ident7ified
# class_tree = np.zeros(np.shape(x_new))
# class_tree[air_index] = 1
# class_tree[tree_index] = 2
# class_tree[iron_index] = 3
# class_tree[lead_index] = 4

# # Create the figure to show the different classes of materials
# fig, ax = plt.subplots()
# cmap = plt.get_cmap('viridis', 4) # Define the colormap
# im = ax.imshow(class_tree, cmap=cmap, vmin=0.5, vmax=4.5) # Plot the image
# cbar = fig.colorbar(im, ticks=[1, 2, 3, 4]) # Define colorbar
# cbar.ax.set_yticklabels(['Air', 'Tree', 'Iron', 'Lead']) # Define tick labels
# ax.set_title(f'No. of rays: {p}\nNo. of angles: {angle_no}')



# plt.show()










