import matplotlib.pyplot as plt
import os
import skimage
from scipy import ndimage, signal
import numpy as np
import time
from tensor_solve import *
import cv2


path = "optical_flow/video_good.mp4"
scalefactor = 10
sigma = 2
N = 5

start = time.time()

video = cv2.VideoCapture(path)
w = int(video.get(3))
h = int(video.get(4))
dim_t = int(video.get(7))
im_3d = np.zeros((h, w, dim_t)) 
im_3d_downscaled = np.zeros((h//scalefactor, w//scalefactor, dim_t))

for i in range(dim_t):
    _, image = video.read()
    im_gray = skimage.color.rgb2gray(image)
    im_gray_downscaled = skimage.transform.downscale_local_mean(im_gray, (scalefactor,scalefactor))
    im_3d[:,:, i] = im_gray
    im_3d_downscaled[:,:, i] = im_gray_downscaled

print(f"It took {time.time()- start} seconds to load images")

Vy = ndimage.gaussian_filter1d(im_3d_downscaled, sigma=sigma, order = 1, axis=0)
Vx = ndimage.gaussian_filter1d(im_3d_downscaled, sigma=sigma, order = 1, axis=1)
Vt = ndimage.gaussian_filter1d(im_3d_downscaled, sigma=sigma, order = 1, axis=2)

vector_field = np.zeros((2, h//scalefactor, w//scalefactor, dim_t))
pos = np.meshgrid([np.arange(0, h, scalefactor)], [np.arange(0, w, scalefactor)])



# for i in range(dim_t):
#     vector_field[:, :, :, i] = tensor_solve(Vx[:,:,i], Vy[:,:,i], Vt[:,:,i], N)

x_list = (pos[0].flatten()*scalefactor).astype(int)
y_list = (pos[1].flatten()*scalefactor).astype(int)

start = time.time()
for t in range(dim_t):
    for i in range(np.size(x_list)):
        N = 7 # N has to be uneven because of the r definition below
        r = int((N-1)/2)
        x0=x_list[i]; y0=y_list[i]

        # print(np.shape(Vy_prewitt))
        Vy_p = Vy[y0-r:y0+r+1, x0-r:x0+r+1, t].flatten()
        Vx_p = Vx[y0-r:y0+r+1, x0-r:x0+r+1, t].flatten()
        Vt_p = Vt[y0-r:y0+r+1, x0-r:x0+r+1, t].flatten()
        A = np.stack((Vy_p, Vx_p))

        sol = np.linalg.lstsq(A.T, -Vt_p, rcond = None)
        vector_field[0, x0-1, y0-1, t] = sol[0][0]
        vector_field[1, x0-1, y0-1, t] = sol[0][1]

print(f"It took {time.time()- start} seconds to also find the vector field")



fig, ax = plt.subplots(figsize = (6,6))
background = ax.imshow(im_3d[:,:,0], cmap = 'gray')
opt_flow = ax.quiver(pos[1], pos[0], vector_field[0,:,:, 0], -vector_field[1,:,:, 0])
plt.pause(1)

for i in range(dim_t):

    # Plot the result
    fig.suptitle(f"Optical Flow - Frame {i+1}")
    background.set_data(im_3d[:,:,i])
    opt_flow.set_UVC(vector_field[0,:,:,i], vector_field[0,:,:,i])

    plt.pause(1/10)
    #plt.savefig(f'Optical_flow/toyOpticalFlow/image_flow_{i}.png', dpi = 120)






