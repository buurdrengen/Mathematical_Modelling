import numpy as np
import matplotlib.pyplot as plt
import time
# from final_func import *
from final_func import *

#%% Change these parameters

angle_no = 180 # Define the number of angles to be used in paralleltomo.
p = 200 # Define the number of rays to be used in paralleltomo.
res = 25 # The picture will be (res x res). pixelwidth is (500/res mm)
vol_pellet = 0.5 # Define how much of the pellet is inside the slice
tree_type = 'beech' # can be 'beech'/'fir'

#%%
# Load start time
start = time.time()

# Define the pure attenuation coefficients
x_air = 0
x_steel = 23.7*1e-2
x_lead = 113*1e-2
if tree_type == 'beech':
    x_tree = 1.20*1e-2
elif tree_type == 'fir':
    x_tree = 1.20*1e-2

# Define the attenuation coefficients that are inside the slice
x_steel_found = vol_pellet*x_steel + (1-vol_pellet)*x_tree
x_lead_found = vol_pellet*x_lead + (1-vol_pellet)*x_lead

# Define the pixel_width to be used for defining classification thresholds in image.
pixel_width = 500/res # [mm]
print(f"the size of the pixels in the image is {pixel_width} mm")

# Find the attenuation coefficients within the tree
air_mean = 0
tree_mean = x_tree
steel_mean = (np.square(2)*x_steel + np.square(pixel_width)*x_tree)/np.square(pixel_width)
lead_mean = (np.square(pixel_width)*x_lead + np.square(pixel_width)*x_tree)/np.square(pixel_width)
# Find the point where the classes have equal probability of being
air_tree_separator = (air_mean + tree_mean)/2
tree_steel_separator = (tree_mean + steel_mean)/2
steel_lead_separator = (steel_mean + lead_mean)/2

final_func(angle_no, 
            p, 
            res, 
            air_tree_separator, 
            tree_steel_separator, 
            steel_lead_separator, 
            class_errors=True,
            sample_size=1)


