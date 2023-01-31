
import matplotlib.pyplot as plt
import os
import skimage
import scipy
import numpy as np

im_3d = np.zeros((256, 256, 64))
for i, image_location in enumerate(os.listdir('Mathematical_Modelling/Optical_flow/toyProblem_F22')):
    image = skimage.io.imread(f"Mathematical_Modelling/Optical_flow/toyProblem_F22/{image_location}")
    im_gray = skimage.color.rgb2gray(image)
    im_3d[:,:, i] = im_gray


for i, image_location in enumerate(os.listdir('Mathematical_Modelling/Optical_flow/toyProblem_F22')):
    image = skimage.io.imread(f"Mathematical_Modelling/Optical_flow/toyProblem_F22/{image_location}")
    im_gray = skimage.color.rgb2gray(image)

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

    # if i < 63:
    #     skimage.io.imshow(Vt[:, :, i])
    #     plt.title(f"frame {range(64)[i]}")

    #     plt.pause(1/24)
    #     plt.clf()

    Vy_prewitt = scipy.ndimage.prewitt(im_gray, axis=0)
    Vx_prewitt = scipy.ndimage.prewitt(im_gray, axis=1)
    # Vx_prewitt = scipy.ndimage.prewitt(im_3d, axis=2)
    
    # skimage.io.imshow(Vx_prewitt)
    # plt.title(f"frame {range(64)[i]}")

    # plt.pause(1/24)
    # plt.clf()
    


