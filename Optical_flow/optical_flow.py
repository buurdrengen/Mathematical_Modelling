
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
        #axx.cla(); axy.cla(); axt.cla()
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
    # idle_prosessing()
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
plot_3_gradients(Vx_prewitt, Vy_prewitt, Vt_prewitt, title = "Gradient with Prewitt Method")

#----------------------------------

# Using the Sobel method
Vy_sobel = ndimage.sobel(im_3d, axis=0)
Vx_sobel = ndimage.sobel(im_3d, axis=1)
Vt_sobel = ndimage.sobel(im_3d, axis=2)

# Displaying the gradient
plot_3_gradients(Vx_sobel,Vy_sobel,Vt_sobel, title = "Gradient with Sobel Method")


"""
Problem 2.3: Gaussian Gradient Filters
"""

