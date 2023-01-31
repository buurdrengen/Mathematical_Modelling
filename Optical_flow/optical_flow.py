
import matplotlib.pyplot as plt
import os
import skimage
from skimage import io, color
import scipy
from scipy import ndimage
import numpy as np

im_3d = np.zeros((256, 256, 64))
for i, image_location in enumerate(np.sort(os.listdir('Optical_flow/toyProblem_F22'))):
    image = io.imread(f"Optical_flow/toyProblem_F22/{image_location}")
    im_gray = color.rgb2gray(image)
    im_3d[:,:, i] = im_gray


for i, image_location in enumerate(np.sort(os.listdir('Optical_flow/toyProblem_F22'))):
    image = io.imread(f"Optical_flow/toyProblem_F22/{image_location}")
    im_gray = color.rgb2gray(image)

    """
    Warmup: Making the video
    """
    # skimage.io.imshow(im_gray)
    # plt.title(f"frame {range(64)[i]}")

    # plt.pause(1/24)
    # plt.clf()

    """
    Problem 2.1: low level gradient
    """
    Vy = im_gray[1:, :] - im_gray[:-1, :]
    Vx = im_gray[:, 1:] - im_gray[:, :-1]
    Vt = im_3d[:, :, 1:] - im_3d[:, :, :-1]

    if i < 63:
        skimage.io.imshow(Vt[:, :, i], vmin = -1, vmax = 1)
        plt.title(f"frame {i+1}")

        plt.pause(1/24)
        plt.clf()

    Vy_prewitt = ndimage.prewitt(im_gray, axis=0)
    Vx_prewitt = ndimage.prewitt(im_gray, axis=1)
    #Vt_prewitt = ndimage.prewitt(im_3d, axis=2)
    
    # io.imshow(Vy_prewitt)
    # plt.title(f"frame {i}")

    # plt.pause(1/24)
    # plt.clf()
    


