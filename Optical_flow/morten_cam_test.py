import numpy as np
import skimage
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2
import time


# Conditions
N = 7   # Same N as other script...
sf = 20 # Scale factor for optical flow. Lower is better but slower

amplifier = 10 # Enlargement of vector arrows

r = (N-1)//2

# Loading camera
cam = cv2.VideoCapture(0)


testframe = np.ones((480,640)); downscaled_image_old = skimage.transform.downscale_local_mean(testframe,(sf,sf))
fig = plt.figure(figsize=(12,9))
ax = plt.imshow(testframe, figure=fig, cmap="gray", vmin = 0, vmax = 1)
fig.suptitle("Camera")

N_im = 1000

pos = np.mgrid[0:640:sf,0:480:sf]
vector_field = np.zeros((2,640//sf,480//sf))
x_list = pos[0].flatten()//sf
y_list = pos[1].flatten()//sf


opt_flow = plt.quiver(pos[0], pos[1], vector_field[0], -vector_field[1], figure = fig, color="blue")
plt.show(block = False)
plt.pause(1)

for i in range(N_im):
    start = time.time()
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    
    img = skimage.color.rgb2gray(frame)
    downscaled_image = skimage.transform.downscale_local_mean(img,(sf,sf))
    Vy = ndimage.sobel(downscaled_image, axis=0)
    Vx = ndimage.sobel(downscaled_image, axis=1)
    Vt = downscaled_image - downscaled_image_old

    for j in range(np.size(x_list)):
        x0 = x_list[j]; y0 = y_list[j]
        Vx_p = Vx[y0-r:y0+r+1, x0-r:x0+r+1].flatten()
        Vy_p = Vy[y0-r:y0+r+1, x0-r:x0+r+1].flatten()
        Vt_p = Vt[y0-r:y0+r+1, x0-r:x0+r+1].flatten()

        A = np.stack((Vx_p,Vy_p))

        sol = np.linalg.lstsq(A.T, -Vt_p, rcond = None)
        vector_field[0, x0, y0] = sol[0][0]
        vector_field[1, x0, y0] = sol[0][1]

    
    # # Lets remove small values
    # amplitude_field = vector_field[0]**2 + vector_field[1]**2
    # neglect_value = 0
    # vector_field[0][amplitude_field <= neglect_value] = 0
    # vector_field[1][amplitude_field <= neglect_value] = 0

    # Update Plot
    ax.set_data(img)
    opt_flow.set_UVC(-vector_field[1]*amplifier, -vector_field[0]*amplifier)
    plt.draw()
    downscaled_image_old = downscaled_image
    print(f"_Frametime: {int(np.ceil(1000*(time.time() - start)))}ms", end="\r")

print("Releasing Camera")
cam.release()