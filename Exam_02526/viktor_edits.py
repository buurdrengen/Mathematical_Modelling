import numpy as np
import matplotlib.pyplot as plt
import time
from final_func import final_func

#%% Change these parameters

angle_no = 90 # Define the number of angles to be used in paralleltomo.
p = 110 # Define the number of rays to be used in paralleltomo.
res = 50 # The picture will be (res x res). pixelwidth is (500/res mm)
vol_pellet = 0.5 # Define how much of the pellet is inside the slice
tree_type = 'Fir' # can be 'beech'/'fir'
tree_ring_no = 10
noise_lim = [2.5e-10, 1e-1]

#%%
# Load start time
start = time.time()

# Define the pure attenuation coefficients
x_air = 0
x_steel = 64.5904*1e-2
x_lead = 342.616*1e-2
if tree_type == 'Beech':
    x_tree = 0.2531236*1e-2
elif tree_type == 'Fir':
    x_tree = 0.167049375*1e-2

# Define the attenuation coefficients that are inside the slice
x_steel_found = vol_pellet*x_steel + (1-vol_pellet)*x_tree
x_lead_found = vol_pellet*x_lead + (1-vol_pellet)*x_tree

# Define the pixel_width to be used for defining classification thresholds in image.
pixel_width = 500/res # [mm]
print(f"the size of the pixels in the image is {pixel_width} mm")

# Find the attenuation coefficients within the tree
air_mean = 0
tree_mean = x_tree*pixel_width/0.1
steel_mean = (np.square(2)*x_steel + np.square(pixel_width-2)*x_tree)/np.square(pixel_width)*pixel_width/0.1
lead_mean = (np.square(2)*x_lead + np.square(pixel_width-2)*x_tree)/np.square(pixel_width)*pixel_width/0.1

# Find the point where the classes have equal probability of being
air_tree_separator = (air_mean + tree_mean)/2
tree_steel_separator = (tree_mean + steel_mean)/2
steel_lead_separator = (steel_mean + lead_mean)/2

print([air_mean, tree_mean, steel_mean, lead_mean])
print([air_tree_separator, tree_steel_separator, steel_lead_separator])


final_func(angle_no, 
            p, 
            res, 
            air_tree_separator, 
            tree_steel_separator, 
            steel_lead_separator, 
            class_errors=True,
            sample_size=20,
            tree_type=tree_type,
            noise_limit=noise_lim,
            ring_count = tree_ring_no,
            vol_pellet = vol_pellet
            )

plt.show()

