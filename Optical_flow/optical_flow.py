
import matplotlib.pyplot as plt
import os
import skimage
#from skimage import io, color
import scipy
from scipy import ndimage
import numpy as np


def plot_3_gradients(Vx, Vy, Vt, cmap = "seismic", FPS = 24, title = "Gradient"):

    # Set sensible colormap scale
    vmin = np.min([np.min(Vy),np.min(Vx),np.min(Vt)])
    vmax = np.max([np.max(Vy),np.max(Vx),np.max(Vt)])
    # Account for diverging array size in Vt 
    N_im = np.min([np.shape(Vx)[2],np.shape(Vy)[2],np.shape(Vt)[2]])
    if N_im == 0: return None

    # Set up figure parameters
    fig, [axx,axy,axt] = plt.subplots(1, 3, figsize=(14, 4))
    cb_ax = fig.add_axes([0.91, 0.1, 0.02, 0.8])

    # Initiate the figure
    axx.set_title("Gradient x"); axy.set_title("Gradient y"); axt.set_title("Gradient t")
    imx = axx.imshow(Vx[:,:,0], vmin = vmin, vmax = vmax, cmap = cmap)
    imy = axy.imshow(Vy[:,:,0], vmin = vmin, vmax = vmax, cmap = cmap)
    imt = axt.imshow(Vt[:,:,0], vmin = vmin, vmax = vmax, cmap = cmap)
    fig.colorbar(mappable=imt, cax=cb_ax)

    # Iterate the plot
    for i in range(N_im):
        fig.suptitle(f"{title} - Frame {i+1}")
        imx.set_data(Vx[:,:,i])
        imy.set_data(Vy[:,:,i])
        imt.set_data(Vt[:,:,i])
        plt.pause(1/FPS)

    plt.close(fig)


# ---------------------------------------------------------------

"""
Problem 1.1: Making the video
"""

# Loading all 64 images into a 3D array as grayscale


path = os.getcwd()
if path[path.rfind("/")+1:] == "Optical_flow":
    new_path = path[:path.rfind("/")]
    os.chdir(new_path)

image_name_list = np.sort(os.listdir('Optical_flow/toyProblem_F22'))
N_im = np.size(image_name_list)
im_3d = np.zeros((256, 256, N_im))

for i, image_location in enumerate(image_name_list):
    image = skimage.io.imread(f"Optical_flow/toyProblem_F22/{image_location}")
    im_gray = skimage.color.rgb2gray(image)
    im_3d[:,:, i] = im_gray

# Displaying the image sequense
imm = skimage.io.imshow(im_3d[:,:,0])
for i in range(N_im):
    imm.set_data(im_3d[:,:,i])
    plt.title(f"Frame {i+1}")
    plt.pause(1/24)

plt.close()

"""
Problem 2.1: Low Level Gradient
"""

# Computing Vx, Vy and Vt
Vy = im_3d[1:, :, :] - im_3d[:-1, :, :]
Vx = im_3d[:, 1:, :] - im_3d[:, :-1, :]
Vt = im_3d[:, :, 1:] - im_3d[:, :, :-1]

plot_3_gradients(Vx, Vy, Vt, title = "Crude Gradient")

"""
Problem 2.2: Simple Gradient Filters
"""


# Using the Prewitt method

Vy_prewitt = ndimage.prewitt(im_3d, axis=0)
Vx_prewitt = ndimage.prewitt(im_3d, axis=1)
Vt_prewitt = ndimage.prewitt(im_3d, axis=2)

# Displaying the gradient
plot_3_gradients(Vx_prewitt, Vy_prewitt, Vt_prewitt, title = "Gradient with Prewitt Filter")

#----------------------------------

# Using the Sobel method
Vy_sobel = ndimage.sobel(im_3d, axis=0)
Vx_sobel = ndimage.sobel(im_3d, axis=1)
Vt_sobel = ndimage.sobel(im_3d, axis=2)

# Displaying the gradient
plot_3_gradients(Vx_sobel, Vy_sobel, Vt_sobel, title = "Gradient with Sobel Filter")


"""
Problem 2.3: Gaussian Gradient Filters
"""
# Using the Gaussian Kernel

sigma = 1

Vy_gauss = ndimage.gaussian_filter1d(im_3d, sigma=sigma, order = 1, axis=0)
Vx_gauss = ndimage.gaussian_filter1d(im_3d, sigma=sigma, order = 1, axis=1)
Vt_gauss = ndimage.gaussian_filter1d(im_3d, sigma=sigma, order = 1, axis=2)

# Displaying the gradient
plot_3_gradients(Vx_gauss, Vy_gauss, Vt_gauss, title = "Gradient with Gaussian Filter")


#Let's plot all the gradients in one plot to compare.
test_frame = 25
vmin = -1; vmax = 1; cmap = "seismic"

imm = skimage.io.imshow(im_3d[:,:,0])
fig, ax = plt.subplots(3, 4, figsize=(10, 8), constrained_layout=True, sharex = True, sharey=True)
#cb_ax = fig.add_axes([0.97, 0.1, 0.01, 0.8])

# Initiate the figure
#crude gradient
km = ax.flat[0].imshow(Vx[:,:,test_frame]/np.max(Vx[:,:,test_frame]), vmin = vmin, vmax = vmax, cmap = cmap)
ax.flat[0].set_title("Crude Gradient")
ax.flat[0].set_ylabel("Gardient x")
ax.flat[4].imshow(Vy[:,:,test_frame]/np.max(Vy[:,:,test_frame]), vmin = vmin, vmax = vmax, cmap = cmap)
ax.flat[4].set_ylabel("Gardient y")
ax.flat[8].imshow(Vt[:,:,test_frame]/np.max(Vt[:,:,test_frame]), vmin = vmin, vmax = vmax, cmap = cmap)
ax.flat[8].set_ylabel("Gardient t")

# Prewitt Filter
ax.flat[1].imshow(Vx_prewitt[:,:,test_frame]/np.max(Vx_prewitt[:,:,test_frame]), vmin = vmin, vmax = vmax, cmap = cmap)
ax.flat[1].set_title("Prewitt Filter ")
ax.flat[5].imshow(Vy_prewitt[:,:,test_frame]/np.max(Vy_prewitt[:,:,test_frame]), vmin = vmin, vmax = vmax, cmap = cmap)
ax.flat[9].imshow(Vt_prewitt[:,:,test_frame]/np.max(Vt_prewitt[:,:,test_frame]), vmin = vmin, vmax = vmax, cmap = cmap)

# Sobel Filter
km = ax.flat[2].imshow(Vx_sobel[:,:,test_frame]/np.max(Vx_sobel[:,:,test_frame]), vmin = vmin, vmax = vmax, cmap = cmap)
ax.flat[2].set_title("Sobel Filter")
ax.flat[6].imshow(Vy_sobel[:,:,test_frame]/np.max(Vy_sobel[:,:,test_frame]), vmin = vmin, vmax = vmax, cmap = cmap)
ax.flat[10].imshow(Vt_sobel[:,:,test_frame]/np.max(Vt_sobel[:,:,test_frame]), vmin = vmin, vmax = vmax, cmap = cmap)

# Gaussian Filter
km = ax.flat[3].imshow(Vx_gauss[:,:,test_frame]/np.max(Vx_gauss[:,:,test_frame]), vmin = vmin, vmax = vmax, cmap = cmap)
ax.flat[3].set_title("Gaussian Filter")
ax.flat[7].imshow(Vy_gauss[:,:,test_frame]/np.max(Vy_gauss[:,:,test_frame]), vmin = vmin, vmax = vmax, cmap = cmap)
ax.flat[11].imshow(Vt_gauss[:,:,test_frame]/np.max(Vt_gauss[:,:,test_frame]), vmin = vmin, vmax = vmax, cmap = cmap)

#fig.colorbar(mappable=km, cax=cb_ax)
fig.suptitle(f"Normalized Gradients - Frame {test_frame}")
fig.savefig("Compare_gradients.png", dpi = 300, format = "png")

