import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import time
from paralleltomo import * # [A,theta,p,d] = paralleltomo(N,theta,p,d)

# Load a seed for rng so that answers are reproducible and a time to test script
np.random.seed(1)
start = time.time()

# Load the image with lead and steel shot
data = np.array(scipy.io.loadmat("Exam_02526/testImage.mat")['im']) #Pixel size: 0.1 mm, 5000x5000 pixels

# Load a resolution list
res_list = np.array([200])
N_outer = np.size(res_list)

# Create a meshgrid for the internal colormaps
resd = np.arange(1,50)
theta_res_list = resd # np.array([1, 2, 3, 4, 5, 6])
p_list = (resd + 1)*3 # np.array([30,40,50, 60, 70, 80, 90, 100])
meshgrid_inner = np.meshgrid(theta_res_list, p_list)
theta_mesh = meshgrid_inner[0].flatten()
p_mesh = meshgrid_inner[1].flatten()
N_inner = np.size(p_mesh)


matrix_cond = np.zeros((N_outer, N_inner))
matrix_cond[:,:] = np.nan

for i in range(N_outer):
    # Define the settings for this configuration
    res_downscale = res_list[i]
    
    # Load the x parameter
    downsized_im = np.copy(data[::res_downscale, ::res_downscale]) # pixel size: [mm] 0.1*res_downscale 
    x = np.copy(downsized_im.flatten())
    N = downsized_im.shape[0]


    print(f"\nWe are now using resolution of ({N}x{N})")

    for j in range(N_inner):
        # Initialize inner loops parameters
        theta_res = theta_mesh[j]
        theta = np.linspace(0,180,max(theta_res)) #np.array([np.arange(0, 180, theta_res)])
        p = p_mesh[j]

        # Initialize system matrix
        [A,_,_,_] = paralleltomo(N, theta, p)
        sensitivity = np.linalg.cond(A)
        print(f"The condition number of our matrix A is given as {sensitivity}. Time passed {np.round(time.time()- start, 2)} seconds")

        # Load the condition number into the predefined matrix
        matrix_cond[i, j] = sensitivity 
# Reshape the matrix so that the dimensions of the matrix correspond to the dimensions of the variables
matrix_cond = np.reshape(matrix_cond, (np.size(res_list), np.size(p_list), np.size(theta_res_list)))

fig, ax = plt.subplots(figsize = [4*res_list.size,4], ncols=res_list.size, sharex=True, layout='constrained')
tick_scale_factor = 4

for i in range(res_list.size):
    #Create image for each resolution
    if res_list.size == 1 : ax = [ax] 
    im=ax[i].imshow(matrix_cond[i,:,:], vmin=0, vmax=1e3, cmap="ocean")
    
    # Define x- and yticks and set the resolution label
    ax[i].set_xticks(range(theta_res_list.size)[::tick_scale_factor], (180/theta_res_list).astype(int)[::tick_scale_factor])
    ax[i].set_title(f"Res: ({int(5000/res_list[i])} x {int(5000/res_list[i])})")
    ax[i].set_yticks(range(p_list.size)[::tick_scale_factor], p_list[::tick_scale_factor])
    ax[i].set_xlabel('# of angles')
fig.suptitle('Sensitivity Study')# Set title
ax[0].set_ylabel(f"No. of Rays") # Create the resolution label


# Colorbar. This can be shared because an upper and lower limit on the image is defined.
fig.colorbar(im, ax=ax)
plt.draw()
plt.savefig("N25-plot",dpi=300,format="png")
plt.show()










