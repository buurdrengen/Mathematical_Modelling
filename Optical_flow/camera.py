import numpy as np
import skimage
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2
import time
from tensor_solve import *


# Conditions
N = 5   # Same N as other script...
scale_factor = 4 # Scale factor for optical flow. Lower is better but slower
figsize = (9,16)
N_a = 8 # Distance between arrows
sigma = 3

image_source = "Optical_flow/Videos/Good_test_video.mp4"

r = (N-1)//2

# Loading camera
cam = cv2.VideoCapture(image_source)

w = int(cam.get(3))
h = int(cam.get(4))
n_frames = int(cam.get(7))




test_frame = np.random.rand(h,w,3)
frame = np.copy(test_frame[:,:,0]); downscaled_image_old = skimage.transform.downscale_local_mean(frame,(scale_factor,scale_factor)); downscaled_image_new = np.copy(downscaled_image_old); downscaled_image = np.copy(downscaled_image_old)
fig, ax = plt.subplots(1, 1, figsize = figsize)
background = plt.imshow(test_frame)
fig.suptitle("Camera")

N_im = (n_frames-1)*(n_frames > 0) + (n_frames == -1)*100

ret, frame = cam.read()

pos = np.mgrid[0:h:scale_factor,0:w:scale_factor]
vector_field = np.zeros((2,h//scale_factor,w//scale_factor))


opt_flow = plt.quiver(pos[1,::N_a,::N_a], pos[0,::N_a,::N_a], vector_field[0,::N_a,::N_a], vector_field[1,::N_a,::N_a], vector_field[0,::N_a,::N_a], cmap = "hot")


for i in range(N_im):
    start = time.time()
    ret, new_frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    
    img = skimage.color.rgb2gray(new_frame)
    downscaled_image_new = skimage.transform.downscale_local_mean(img,(scale_factor,scale_factor))

    image_stack = np.stack((downscaled_image_old, downscaled_image, downscaled_image_new))
    # Shape: (3,y,x)
    Vy = ndimage.sobel(image_stack, axis=1)[1,:,:]
    Vx = ndimage.sobel(image_stack, axis=2)[1,:,:]
    Vt = ndimage.sobel(image_stack, axis=0)[1,:,:]
    
    # Vy = ndimage.gaussian_filter1d(image_stack, sigma = sigma, order = 1, axis=1)[1,:,:]
    # Vx = ndimage.gaussian_filter1d(image_stack, sigma = sigma, order = 1, axis=2)[1,:,:]
    # Vt = ndimage.gaussian_filter1d(image_stack, sigma = sigma, order = 1, axis=0)[1,:,:]


    vector_field[:,:,:] = tensor_solve(Vx = Vx, Vy = Vy, Vt = Vt, N = N)

    #print(tensor_solve(Vx = Vx.T, Vy = Vy.T, Vt = Vt.T, N = N))
    # for j in range(np.size(x_list)):
    #     # Try to implement np.tensordot
    #     x0 = x_list[j]; y0 = y_list[j]
    #     u1 = x0-r; u2 = x0+r+1; 
    #     v1 = y0-r; v2 = y0+r+1
    #     if u1 == 0: u1 = None
    #     if v1 == 0: v1 = None

    #     Vx_p = Vx[v1:v2, u1:u2].flatten()
    #     Vy_p = Vy[v1:v2, u1:u2].flatten()
    #     Vt_p = Vt[v1:v2, u1:u2].flatten()

    #     A = np.stack((Vx_p,Vy_p))

    #     sol = np.linalg.lstsq(A.T, -Vt_p, rcond=None)
    #     vector_field[0, x0, y0] = sol[0][0]
    #     vector_field[1, x0, y0] = sol[0][1]

    
    # Lets remove small values
    amplitude_field = vector_field[0,::N_a,::N_a]**2 + vector_field[1,::N_a,::N_a]**2
    # neglect_value = 0.05
    # vector_field[0,::N_a,::N_a][amplitude_field <= neglect_value] = 0
    # vector_field[1,::N_a,::N_a][amplitude_field <= neglect_value] = 0

    # Update Plot
    background.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    opt_flow.set_UVC(vector_field[0,::N_a,::N_a], -vector_field[1,::N_a,::N_a], amplitude_field)
    plt.pause(0.01)
    
    downscaled_image_old = np.copy(downscaled_image)
    downscaled_image = np.copy(downscaled_image_new)
    frame = np.copy(new_frame)



    #plt.savefig(f'Optical_flow/VideoFlow/GV_flow_{i}.png', dpi = 70)

    print(f"_Frametime: {int(np.ceil(1000*(time.time() - start)))}ms, max movement: {np.round(np.sqrt(np.max(amplitude_field)),2)}, zero-points: {(vector_field[:,:,:] == 0).sum()}                          ", end="\r")

print("\nReleasing Camera")
cam.release()