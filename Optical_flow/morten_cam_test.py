import numpy as np
import skimage
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2
import time
from tensor_solve import *


# Conditions
N = 5   # Same N as other script...
sf = 5 # Scale factor for optical flow. Lower is better but slower

r = (N-1)//2

# Loading camera
cam = cv2.VideoCapture(0)


test_frame = np.random.rand(480,640,3); frame = np.copy(test_frame[:,:,0]); downscaled_image_old = skimage.transform.downscale_local_mean(frame,(sf,sf)); downscaled_image_new = np.copy(downscaled_image_old); downscaled_image = np.copy(downscaled_image_old)
fig = plt.figure(figsize=(8,6))
ax = plt.imshow(test_frame, figure=fig)
fig.suptitle("Camera")

N_im = 1000

pos = np.mgrid[0:640:sf,0:480:sf]
vector_field = np.ones((2,640//sf,480//sf))
x_list = pos[0].flatten()//sf
y_list = pos[1].flatten()//sf

ret, frame = cam.read()

opt_flow = plt.quiver(pos[0], pos[1], vector_field[0], vector_field[1], figure = fig, scale = 20)

for i in range(N_im):
    start = time.time()
    ret, new_frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    
    img = skimage.color.rgb2gray(new_frame)
    downscaled_image_new = skimage.transform.downscale_local_mean(img,(sf,sf))

    image_stack = np.stack((downscaled_image_old, downscaled_image, downscaled_image_new))

    Vy = ndimage.sobel(image_stack, axis=0)[1,:,:]
    Vx = ndimage.sobel(image_stack, axis=1)[1,:,:]
    #Vt = ndimage.sobel(image_stack, axis=2)[1,:,:]
    Vt = downscaled_image - downscaled_image_old

    for j in range(np.size(x_list)):
        # Try to implement np.tensordot
        x0 = x_list[j]; y0 = y_list[j]
        u1 = x0-r; u2 = x0+r+1; 
        v1 = y0-r; v2 = y0+r+1
        if u1 == 0: u1 = None
        if v1 == 0: v1 = None

        Vx_p = Vx[v1:v2, u1:u2].flatten()
        Vy_p = Vy[v1:v2, u1:u2].flatten()
        Vt_p = Vt[v1:v2, u1:u2].flatten()

        A = np.stack((Vx_p,Vy_p))

        sol = np.linalg.lstsq(A.T, -Vt_p, rcond=None)
        vector_field[0, x0, y0] = sol[0][0]
        vector_field[1, x0, y0] = sol[0][1]

    
    # Lets remove small values
    amplitude_field = vector_field[0]**2 + vector_field[1]**2
    neglect_value = 0.05
    vector_field[0][amplitude_field <= neglect_value] = 0
    vector_field[1][amplitude_field <= neglect_value] = 0

    # Update Plot
    ax.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    opt_flow.set_UVC(vector_field[0], vector_field[1])
    plt.pause(0.01)
    
    downscaled_image_old = np.copy(downscaled_image)
    downscaled_image = np.copy(downscaled_image_new)
    frame = np.copy(new_frame)

    print(f"_Frametime: {int(np.ceil(1000*(time.time() - start)))}ms, max movement: {np.round(np.sqrt(np.max(amplitude_field)),2)}                           ", end="\r")

print("\nReleasing Camera")
cam.release()