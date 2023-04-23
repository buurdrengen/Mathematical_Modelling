import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import time
import paralleltomo # [A,theta,p,d] = paralleltomo(N,theta,p,d)

# Load a seed for rng so that answers are reproducible and a time to test script
np.random.seed(1)
start = time.time()

# Load the image with lead and steel shot
data = np.array(scipy.io.loadmat("Exam_02526/testImage.mat")['im']) #Pixel size: 0.1 mm, 5000x5000 pixels

# Load a resolution list
res_list = np.array([100,200])
N_outer = np.size(res_list)

# Create a meshgrid for the internal colormaps
theta_res_list = np.array([1, 2, 3])
p_list = np.array([50, 60])#, 70, 80, 90, 100])
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
        theta = np.array([np.arange(0, 180, theta_res)])
        p = p_mesh[j]

        # Initialize system matrix
        [A,_,_,_] = paralleltomo.paralleltomo(N, theta, p)
        sensitivity = np.linalg.cond(A)
        print(f"The condition number of our matrix A is given as {sensitivity}. Time passed {np.round(time.time()- start, 2)} seconds")

        # Load the condition number into the predefined matrix
        matrix_cond[i, j] = sensitivity 
# Reshape the matrix so that the dimensions of the matrix correspond to the dimensions of the variables
matrix_cond = np.reshape(matrix_cond, (np.size(res_list), np.size(p_list), np.size(theta_res_list)))

fig, ax = plt.subplots(ncols=res_list.size, sharex=True, layout='constrained')

for i in range(res_list.size):
    #Create image for each resolution 
    im=ax[i].imshow(matrix_cond[i,:,:], vmin=0, vmax=1e3)
    
    # Define x- and yticks and set the resolution label
    ax[i].set_xticks(range(theta_res_list.size), (180/theta_res_list).astype(int))
    ax[i].set_title(f"Res: ({int(5000/res_list[i])} x {int(5000/res_list[i])})")
    ax[i].set_yticks(range(p_list.size), p_list)
    ax[i].set_xlabel('# of angles')
fig.suptitle('Sensitivity Stude')# Set title
ax[0].set_ylabel(f"No. of Rays") # Create the resolution label


# Colorbar. This can be shared because an upper and lower limit on the image is defined.
fig.colorbar(im, ax=ax)
plt.show()










