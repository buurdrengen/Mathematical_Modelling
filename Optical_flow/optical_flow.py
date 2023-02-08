
import matplotlib.pyplot as plt
import os
import skimage
#from skimage import io, color
from scipy import ndimage, signal
import numpy as np
import time


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

print(" Loading Files...", end="\r")

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
    imm.set_data(im_3d[:,:,i])
    # plt.title(f"Frame {i+1}")
    # plt.pause(1/24)

plt.close()

"""
Problem 2.1: Low Level Gradient
"""

print(" Computing gradients... ", end="\r")

# Computing Vx, Vy and Vt
Vy = im_3d[1:, :, :] - im_3d[:-1, :, :]
Vx = im_3d[:, 1:, :] - im_3d[:, :-1, :]
Vt = im_3d[:, :, 1:] - im_3d[:, :, :-1]

# plot_3_gradients(Vx, Vy, Vt, title = "Crude Gradient")


"""
Problem 2.2: Simple Gradient Filters
"""


# Using the Prewitt method

Vy_prewitt = ndimage.prewitt(im_3d, axis=0)
Vx_prewitt = ndimage.prewitt(im_3d, axis=1)
Vt_prewitt = ndimage.prewitt(im_3d, axis=2)

# Displaying the gradient
# plot_3_gradients(Vx_prewitt, Vy_prewitt, Vt_prewitt, title = "Gradient with Prewitt Filter")

#----------------------------------

# Using the Sobel method
Vy_sobel = ndimage.sobel(im_3d, axis=0)
Vx_sobel = ndimage.sobel(im_3d, axis=1)
Vt_sobel = ndimage.sobel(im_3d, axis=2)

# Displaying the gradient
# plot_3_gradients(Vx_sobel, Vy_sobel, Vt_sobel, title = "Gradient with Sobel Filter")


"""
Problem 2.3: Gaussian Gradient Filters
"""
# Using the Gaussian Kernel

sigma = 2

Vy_gauss = ndimage.gaussian_filter1d(im_3d, sigma=sigma, order = 1, axis=0)
Vx_gauss = ndimage.gaussian_filter1d(im_3d, sigma=sigma, order = 1, axis=1)
Vt_gauss = ndimage.gaussian_filter1d(im_3d, sigma=sigma, order = 1, axis=2)

# Displaying the gradient
# plot_3_gradients(Vx_gauss, Vy_gauss, Vt_gauss, title = "Gradient with Gaussian Filter")


#Let's plot all the gradients in one plot to compare.
test_frame = 25
vmin = -1; vmax = 1; cmap = "seismic"

imm = skimage.io.imshow(im_3d[:,:,0])
fig, ax = plt.subplots(3, 4, figsize=(10, 8), constrained_layout=True, sharex = True, sharey=True)
#cb_ax = fig.add_axes([0.97, 0.1, 0.01, 0.8])

# Initiate the figure
#crude gradient
km = ax.flat[0].imshow(Vx[:,:,test_frame]/np.max(Vx[:,:,test_frame]), vmin = vmin, vmax = vmax, cmap = cmap)
ax.flat[0].set_title("Crude Gradient")
ax.flat[0].set_ylabel("Gardient x")
ax.flat[4].imshow(Vy[:,:,test_frame]/np.max(Vy[:,:,test_frame]), vmin = vmin, vmax = vmax, cmap = cmap)
ax.flat[4].set_ylabel("Gardient y")
ax.flat[8].imshow(Vt[:,:,test_frame]/np.max(Vt[:,:,test_frame]), vmin = vmin, vmax = vmax, cmap = cmap)
ax.flat[8].set_ylabel("Gardient t")

# Prewitt Filter
ax.flat[1].imshow(Vx_prewitt[:,:,test_frame]/np.max(Vx_prewitt[:,:,test_frame]), vmin = vmin, vmax = vmax, cmap = cmap)
ax.flat[1].set_title("Prewitt Filter ")
ax.flat[5].imshow(Vy_prewitt[:,:,test_frame]/np.max(Vy_prewitt[:,:,test_frame]), vmin = vmin, vmax = vmax, cmap = cmap)
ax.flat[9].imshow(Vt_prewitt[:,:,test_frame]/np.max(Vt_prewitt[:,:,test_frame]), vmin = vmin, vmax = vmax, cmap = cmap)

# Sobel Filter
km = ax.flat[2].imshow(Vx_sobel[:,:,test_frame]/np.max(Vx_sobel[:,:,test_frame]), vmin = vmin, vmax = vmax, cmap = cmap)
ax.flat[2].set_title("Sobel Filter")
ax.flat[6].imshow(Vy_sobel[:,:,test_frame]/np.max(Vy_sobel[:,:,test_frame]), vmin = vmin, vmax = vmax, cmap = cmap)
ax.flat[10].imshow(Vt_sobel[:,:,test_frame]/np.max(Vt_sobel[:,:,test_frame]), vmin = vmin, vmax = vmax, cmap = cmap)

# Gaussian Filter
km = ax.flat[3].imshow(Vx_gauss[:,:,test_frame]/np.max(Vx_gauss[:,:,test_frame]), vmin = vmin, vmax = vmax, cmap = cmap)
ax.flat[3].set_title("Gaussian Filter")
ax.flat[7].imshow(Vy_gauss[:,:,test_frame]/np.max(Vy_gauss[:,:,test_frame]), vmin = vmin, vmax = vmax, cmap = cmap)
ax.flat[11].imshow(Vt_gauss[:,:,test_frame]/np.max(Vt_gauss[:,:,test_frame]), vmin = vmin, vmax = vmax, cmap = cmap)

#fig.colorbar(mappable=km, cax=cb_ax)
fig.suptitle(f"Normalized Gradients - Frame {test_frame}")
fig.savefig("Compare_gradients.png", dpi = 300, format = "png")


"""
Problem 3.1
"""
# N = 3 # N has to be uneven because of the r definition below
# r = int((N-1)/2)
# x0=100; y0=100; t0 = 30

# # print(np.shape(Vy_prewitt))
# Vy_p = Vy_prewitt[y0-r:y0+r+1, x0-r:x0+r+1, t0].flatten()
# Vx_p = Vx_prewitt[y0-r:y0+r+1, x0-r:x0+r+1, t0].flatten()
# Vt_p = Vt_prewitt[y0-r:y0+r+1, x0-r:x0+r+1, t0].flatten()
# A = np.stack((Vy_p, Vx_p))

# sol = np.linalg.lstsq(A.T, -Vt_p, rcond = None)
# print(sol[0])


"""
Problem 3.2
"""

# pos = np.mgrid[r:256-r, r:256-r]
# x_list = pos[0,:,:].flatten()
# y_list = pos[1,:,:].flatten()

# vector_field = np.zeros(np.shape(pos))
# start = time.time()
# for i in range(np.size(x_list)):
#     N = 7 # N has to be uneven because of the r definition below
#     r = int((N-1)/2)
#     x0=x_list[i]; y0=y_list[i]; t0 = 30

#     # print(np.shape(Vy_prewitt))
#     Vy_p = Vy_prewitt[y0-r:y0+r+1, x0-r:x0+r+1, t0].flatten()
#     Vx_p = Vx_prewitt[y0-r:y0+r+1, x0-r:x0+r+1, t0].flatten()
#     Vt_p = Vt_prewitt[y0-r:y0+r+1, x0-r:x0+r+1, t0].flatten()
#     A = np.stack((Vy_p, Vx_p))

#     sol = np.linalg.lstsq(A.T, -Vt_p, rcond = None)
#     vector_field[0, x0-1, y0-1] = sol[0][0]
#     vector_field[1, x0-1, y0-1] = sol[0][1]
# print(f"There passed {np.round((time.time()-start), 3)} seconds")

# plt.close()
# plt.figure()
# plt.imshow(im_3d[:,:,30], cmap = 'gray') 
# plt.quiver(pos[0,::10,::10], pos[1,::10,::10], vector_field[0,::10,::10], vector_field[1,::10,::10])
# plt.show()


########### GIF in 3.2
N = 9 # N has to be uneven because of the r definition below
r = int((N-1)/2)
tmax = N_im
pos = np.mgrid[r:256-r, r:256-r, 0:tmax]
x_list = pos[0].flatten()
y_list = pos[1].flatten()
t_list = pos[2].flatten()

vector_field = np.zeros((2, 256-2*r, 256-2*r, tmax))
start = time.time()

print("Computing flow...      ")
for i in range(np.size(x_list)):
    if i%1e4 == 0: print(f" Operations: {np.round(i*1e-6,2)} million ", end="\r")
    x0=x_list[i]; y0=y_list[i]; t0 = t_list[i]

    Vy_p = Vy_sobel[y0-r:y0+r+1, x0-r:x0+r+1, t0].flatten()
    Vx_p = Vx_sobel[y0-r:y0+r+1, x0-r:x0+r+1, t0].flatten()
    Vt_p = Vt_sobel[y0-r:y0+r+1, x0-r:x0+r+1, t0].flatten()
    A = np.stack((Vx_p, Vy_p))

    sol = np.linalg.lstsq(A.T, -Vt_p, rcond = None)
    vector_field[0, x0-r, y0-r, t0] = sol[0][0]
    vector_field[1, x0-r, y0-r, t0] = sol[0][1]

print(f"\nDone in {time.strftime('%-M minutes and %-S seconds', time.gmtime(time.time()-start))}")



#print(np.shape(pos))

plt.close()
plt.show()
plt.close()

N_a = 5 # This is the distance in pixels between each quiver arrow
average_filter =  np.ones([N_a, N_a])/(N_a**2)

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# background_im = ax.imshow(im_3d[:,:,0], cmap = 'gray') 
# opt_flow = ax.quiver(pos[0,::10,::10, 0], pos[1,::10,::10, 0], vector_field[0,::10,::10, 0], vector_field[1,::10,::10, 0])

for i in range(tmax):

    # Lets try to average the vector field
    quiver_field_x = signal.convolve2d(vector_field[0,:,:,i], average_filter, mode = "same")
    quiver_field_y = signal.convolve2d(vector_field[1,:,:,i], average_filter, mode = "same")

    # Lets remove small values
    amplitude_field = quiver_field_x**2 + quiver_field_y**2
    neglect_value = 0.2
    quiver_field_x[amplitude_field <= neglect_value] = 0
    quiver_field_y[amplitude_field <= neglect_value] = 0

    # Plot the result
    fig.suptitle(f"Optical Flow - Frame {i+1}")
    ax.imshow(im_3d[:,:,i], cmap = 'gray') 
    opt_flow = ax.quiver(pos[0,::N_a,::N_a, i], pos[1,::N_a,::N_a, i], quiver_field_x[::N_a,::N_a], -quiver_field_y[::N_a,::N_a])
    # Check the sign of quiver_field_y...
    # It seems to be invereted...
    #ax.arrow(10,100,50,50)
    plt.pause(1/2)
    plt.savefig(f'Optical_flow/toyOpticalFlow/image_flow_{i}.png', dpi = 120)
    plt.cla()
