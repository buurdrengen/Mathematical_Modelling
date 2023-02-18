import numpy as np
import skimage
import matplotlib.pyplot as plt
import matplotlib.animation as anim
# import matplotlib.quiver
from scipy import ndimage
import cv2
import time
from tensor_solve import *


# Conditions
N = 7   # Same N as other script...
scale_factor = 4 # Scale factor for optical flow. Lower is better but slower
figsize = (8,4.5)
N_a = 8 # Distance between arrows
sigma = 4

fps = 15
name = "Spinning Wheel"


image_source = "Optical_flow/Videos/Spinny_wheel.mp4"

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


opt_flow = plt.quiver(pos[1,::N_a,::N_a], pos[0,::N_a,::N_a], vector_field[0,::N_a,::N_a], vector_field[1,::N_a,::N_a], vector_field[0,::N_a,::N_a], cmap = "hot", scale = 10000) #


movie_writer = anim.writers['ffmpeg']
metadata = dict(title=name, authors = 'Aksel Buur Christensen, Karen Witness, Morten Westermann, Viktor Isaksen')
movie = movie_writer(fps=fps, metadata=metadata)


with movie.saving(fig, name + ".mp4", n_frames):
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
        Vy = ndimage.gaussian_filter1d(image_stack, axis=1, order = 1, sigma = sigma, mode="constant", cval=0)[1,:,:]
        Vx = ndimage.gaussian_filter1d(image_stack, axis=2, order = 1, sigma = sigma, mode="constant", cval=0)[1,:,:]
        Vt = ndimage.sobel(image_stack, axis=0)[1,:,:]

        vector_field[:,:,:] = tensor_solve(Vx = Vx, Vy = Vy, Vt = Vt, N = N)

        # Lets remove small values
        amplitude_field = np.sqrt(vector_field[0,::N_a,::N_a]**2 + vector_field[1,::N_a,::N_a]**2)

        # Update Plot
        background.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        opt_flow.set_UVC(vector_field[0,::N_a,::N_a], -vector_field[1,::N_a,::N_a], amplitude_field ) #
        plt.draw()
        movie.grab_frame()
        
        downscaled_image_old = np.copy(downscaled_image)
        downscaled_image = np.copy(downscaled_image_new)
        frame = np.copy(new_frame)

        print(f"_ Frame: {i}, Frametime: {int(np.ceil(1000*(time.time() - start)))}ms, max movement: {np.round(np.sqrt(np.max(amplitude_field)),2)}, zero-points: {(vector_field[:,:,:] == 0).sum()}                          ", end="\r")

print("\nReleasing Camera")
cam.release()