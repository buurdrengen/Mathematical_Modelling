import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time
from paralleltomo import * # [A,theta,p,d] = paralleltomo(N,theta,p,d)

# Load a seed for rng so that answers are reproducible and a time to test script
np.random.seed(1)
start = time.time()

# Load the image with lead and steel shot
data = np.array(scipy.io.loadmat("Exam_02526/testImage.mat")['im']) #Pixel size: 0.1 mm, 5000x5000 pixels

# Load a resolution list
res_list = np.array([100])
N_outer = np.size(res_list)

# Create a meshgrid for the internal colormaps
n = 25
theta_res_list = np.linspace(1,180,n) # np.array([1, 2, 3, 4, 5, 6])
p_list = np.arange(1,n)*3 # np.array([30,40,50, 60, 70, 80, 90, 100])
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
        theta = np.array([np.linspace(0,180,int(theta_res))]) #np.array([np.arange(0, 180, theta_res)])
        p = p_mesh[j]

        # Initialize system matrix
        [A,_,_,_] = paralleltomo(N, theta, p)
        sensitivity = np.linalg.cond(A)
        print(f" Completion: {100*j/N_inner:0.2f}%, condition number: {sensitivity}. Time passed {np.round(time.time()- start, 2)}s                ", end="\r")

        # Load the condition number into the predefined matrix
        matrix_cond[i, j] = sensitivity
    print("")
# Reshape the matrix so that the dimensions of the matrix correspond to the dimensions of the variables
matrix_cond = np.reshape(matrix_cond, (np.size(res_list), np.size(p_list), np.size(theta_res_list)))
matrix_cond[matrix_cond > 1e4] = 1e4
matrix_cond[matrix_cond < 1e1] = 1e1


fig, ax = plt.subplots(figsize = [4*res_list.size,4], ncols=res_list.size, sharex=True, layout='constrained')
tick_scale_factor = 4

for i in range(res_list.size):
    #Create image for each resolution
    if res_list.size == 1 : ax = [ax] 
    im=ax[i].imshow(matrix_cond[i,:,:], cmap="ocean", norm=LogNorm(1e1,1e4))
    
    # Define x- and yticks and set the resolution label
    ax[i].set_xticks(range(theta_res_list.size)[::tick_scale_factor], (theta_res_list).astype(int)[::tick_scale_factor])
    ax[i].set_title(f"Res: ({int(5000/res_list[i])} x {int(5000/res_list[i])})")
    ax[i].set_yticks(range(p_list.size)[::tick_scale_factor], p_list[::tick_scale_factor])
    ax[i].set_xlabel('# of angles')
fig.suptitle('Sensitivity Study')# Set title
ax[0].set_ylabel(f"No. of Rays") # Create the resolution label


# Colorbar. This can be shared because an upper and lower limit on the image is defined.
fig.colorbar(im, ax=ax)

mdict = {"matrix_cond": matrix_cond, "label": "matrix_cond"}
scipy.io.savemat("N50-data.mat",mdict)


plt.draw()
plt.savefig("N50-plot.png",dpi=300,format="png")
plt.show()










