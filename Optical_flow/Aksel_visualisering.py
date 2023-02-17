## Aksel optical flow visualisering ## 
import numpy as np 
import matplotlib.pyplot as plt
import os
from skimage.color import rgb2gray
from skimage import io 
import skimage 
import scipy.ndimage

images = os.listdir('Optical_flow/toyProblem_F22')
for i in images: 
    frame = io.imread('Optical_flow/toyProblem_F22/'+i)
    gray_frame = rgb2gray(frame)
    skimage.io.imshow(gray_frame)
    plt.pause(0.01)
    plt.clf()

    Vx_gray = gray_frame[:,1:] - gray_frame[0:,:-1]
    Vy_gray = gray_frame[1:,:] - gray_frame[:-1,:]
    Vt_gray = np.ones((256,256))
    #skimage.io.imshow(Vx_gray)
    #plt.pause(0.01)
    #plt.clf()
    
    prewitx = scipy.ndimage.prewitt(gray_frame,axis=1)
    prewity = scipy.ndimage.prewitt(gray_frame,axis=0)
    #skimage.io.imshow(prewity)
    #plt.pause(0.01)
    #plt.clf()

    ## Gaussian filter ## 
    # gausx = scipy.ndimage.gaussian_filter(Vx_gray,sigma=4)
    # gausy = scipy.ndimage.gaussian_filter(Vy_gray,sigma=4)
    # skimage.io.imshow(gausx)
    # plt.pause(0.01)
    # plt.clf()



 


