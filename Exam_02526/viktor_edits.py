import numpy as np
import matplotlib.pyplot as plt
import time
from final_func import final_func

#%% Change these parameters

angle_no = [40, 80, 160]
p = [55, 110, 210]
res = [25, 50, 100] # - pixelwidth is (500/res mm)
vol_pellet = 0.5 # Define how much of the pellet is inside the slice
tree_types = ['Beech', 'Fir'] # can be 'beech'/'fir'
tree_ring_no = 10
noise_lim = [1e-4, 1e0]

#%%
# Load start time
start = time.time()

# Define the pure attenuation coefficients
x_air = 0
x_steel = 64.5904*1e-2
x_lead = 342.616*1e-2
for tree_type in tree_types:
    if tree_type == 'Beech':
        x_tree = 0.2531236*1e-2
    elif tree_type == 'Fir':
        x_tree = 0.167049375*1e-2

    # Define the attenuation coefficients that are inside the slice
    x_steel_found = vol_pellet*x_steel + (1-vol_pellet)*x_tree
    x_lead_found = vol_pellet*x_lead + (1-vol_pellet)*x_tree

    # Define the pixel_width to be used for defining classification thresholds in image.
    for j in range(3):
        pixel_width = 500/res[j] # [mm]
        print(f"the size of the pixels in the image is {pixel_width} mm")

        # Find the attenuation coefficients within the tree
        air_mean = 0
        tree_mean = x_tree*pixel_width/0.1
        steel_mean = (np.pi*x_steel_found + (np.square(pixel_width)-np.pi)*x_tree)/np.square(pixel_width)*pixel_width/0.1
        lead_mean = (np.pi*x_lead_found + (np.square(pixel_width)-np.pi)*x_tree)/np.square(pixel_width)*pixel_width/0.1

        # Find the point where the classes have equal probability of being
        air_tree_separator = (air_mean + tree_mean)/2
        tree_steel_separator = (tree_mean + steel_mean)/2
        steel_lead_separator = (steel_mean + lead_mean)/2

        print([air_mean, tree_mean, steel_mean, lead_mean])
        print([air_tree_separator, tree_steel_separator, steel_lead_separator])


        final_func(angle_no[j], 
                    p[j], 
                    res[j], 
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
        print(f"Ended simulation in time: {start - time.time():0.1} seconds")
        plt.close('all')

