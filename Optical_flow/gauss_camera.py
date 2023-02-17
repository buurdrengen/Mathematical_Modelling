import numpy as np
import skimage
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2
import time
from tensor_solve import *


# Conditions
scale_factor = 8 # Scale factor for optical flow. Lower is better but slower
figsize = (9,16)
N_a = 4 # Distance between arrows
sigma_1 = 2
sigma_2 = 1

image_source = "Optical_flow/Videos/Spinny_wheel.mp4"

start = time.time()

# Loading camera
cam = cv2.VideoCapture(image_source)

w = int(cam.get(3))
h = int(cam.get(4))
n_frames = int(cam.get(7))

frame_stack = np.zeros((h,w,3,n_frames), dtype = np.int16)
image_stack_downscaled = np.zeros((h//scale_factor,w//scale_factor,n_frames),dtype=np.float32)
vector_field = np.zeros((h//scale_factor,w//scale_factor,n_frames,2),dtype=np.float32)

for i in range(n_frames):
    ret, frame = cam.read()
    frame_stack[:,:,:,i] = np.int16(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # image_stack[:,:,i] = skimage.color.rgb2gray(frame)
    image_stack_downscaled[:,:,i] = skimage.transform.downscale_local_mean(skimage.color.rgb2gray(frame),(scale_factor,scale_factor))


Vx = ndimage.gaussian_filter1d(image_stack_downscaled, sigma = sigma_1, order = 1, axis = 0)
Vy = ndimage.gaussian_filter1d(image_stack_downscaled, sigma = sigma_1, order = 1, axis = 1)
Vt = ndimage.gaussian_filter1d(image_stack_downscaled, sigma = sigma_1, order = 1, axis = 2)

sxx = ndimage.gaussian_filter(Vx*Vx, sigma = sigma_2)
sxy = ndimage.gaussian_filter(Vx*Vy, sigma = sigma_2)
syy = ndimage.gaussian_filter(Vy*Vy, sigma = sigma_2)

sxt = ndimage.gaussian_filter(Vx*Vt, sigma = sigma_2)
syt = ndimage.gaussian_filter(Vy*Vt, sigma = sigma_2)

A = np.zeros((np.size(sxx), 2, 2))
b = -np.zeros((np.size(sxx), 2))

A[:,0,0] = sxx.flatten()
A[:,0,1] = sxy.flatten()
A[:,1,0] = sxy.flatten()
A[:,1,1] = syy.flatten()

b[:,0] = -sxt.flatten()
b[:,1] = -syt.flatten()

sol = np.linalg.solve(A,b)

vector_field[:,:,:,:] = sol.reshape((h//scale_factor,w//scale_factor,n_frames,2))

fig, ax = plt.subplots(1, 1, figsize = figsize)
background = plt.imshow(frame_stack[:,:,:,0])
fig.suptitle("Camera")

amplitude_field = vector_field[::N_a,::N_a,:,0]**2 + vector_field[::N_a,::N_a,:,1]**2

pos = np.mgrid[0:h:scale_factor,0:w:scale_factor]
opt_flow = plt.quiver(pos[1,::N_a,::N_a], pos[0,::N_a,::N_a], vector_field[::N_a,::N_a,0,0], vector_field[::N_a,::N_a,0,1],amplitude_field[:,:,0])

print(f"\nDone in {time.strftime('%-M minutes and %-S seconds', time.gmtime(time.time()-start))}")

for i in range(n_frames):
    # Update Plot
    background.set_data(frame_stack[:,:,:,i])
    opt_flow.set_UVC(vector_field[::N_a,::N_a,i,0], vector_field[::N_a,::N_a,i,1], amplitude_field[:,:,i])
    plt.pause(0.1)
    

print("\nReleasing Camera")
cam.release()