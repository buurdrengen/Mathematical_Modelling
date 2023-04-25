import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import paralleltomo # [A,theta,p,d] = paralleltomo(N,theta,p,d)
import math


# Load the image with lead and steel shot
data = np.array(scipy.io.loadmat("Exam_02526/testImage.mat")['im']) #Pixel size: 0.1 mm
downsized_im_100 = data[::50, ::50] # pixel size: 100x100
downsized_im_50 = data[::100, ::100] # pixel size: 50x50
downsized_im_25 = data[::200, ::200] # pixel size: 25x25
x_100 = np.copy(downsized_im_100.flatten())
x_50 = np.copy(downsized_im_50.flatten())
x_25 = np.copy(downsized_im_25.flatten())

# Settings 100 with P = 100 rays, assuming there has to be a ray for every pixel 
N_100 = downsized_im_100.shape[0]
theta_100 = np.array([np.arange(0, 180, 1)])
p_100 = 100
d_100 = math.sqrt(2)*N_100
[A_100,theta_100,p_100,d_100] = paralleltomo.paralleltomo(N_100, theta_100, p_100)

# Settings 50 with P = 50 rays 
N_50 = downsized_im_50.shape[0]
theta_50 = np.array([np.arange(0, 180, 1)])
p_50 = 50
d_50 = math.sqrt(2)*N_50
[A_50,theta_50,p_50,d_50] = paralleltomo.paralleltomo(N_50, theta_50, p_50)

# Settings 25 with P = 25 rays 
N_25 = downsized_im_25.shape[0]
theta_25 = np.array([np.arange(0, 180, 1)])
p_25 = 25
d_25 = math.sqrt(2)*N_25
[A_25,theta_25,p_25,d_25] = paralleltomo.paralleltomo(N_25, theta_25, p_25)


# Load simulated forward projection 
b_100 = A_100 @ x_100 
b_50 = A_50 @ x_50 # Finding b
b_25 = A_25 @ x_25

# Noise
noise_small = 1e-4
noise_large = 1e-3/2
b_perturbed_noise_small_100 = b_100 + np.random.normal(0, noise_small, size=np.shape(b_100)) # adding noise
b_perturbed_noise_large_100 = b_100 + np.random.normal(0, noise_large, size=np.shape(b_100)) # adding noise
b_perturbed_noise_small_50 = b_50 + np.random.normal(0, noise_small, size=np.shape(b_50)) # adding noise
b_perturbed_noise_large_50 = b_50 + np.random.normal(0, noise_large, size=np.shape(b_50)) # adding noise
b_perturbed_noise_small_25 = b_25 + np.random.normal(0, noise_small, size=np.shape(b_25)) # adding noise
b_perturbed_noise_large_25 = b_25 + np.random.normal(0, noise_large, size=np.shape(b_25)) # adding noise



x_noise_small_100 = np.linalg.solve(A_100.T @ A_100, A_100.T @ b_perturbed_noise_small_100)
x_noise_large_100 = np.linalg.solve(A_100.T @ A_100, A_100.T @ b_perturbed_noise_large_100)
x_noise_small_50 = np.linalg.solve(A_50.T @ A_50, A_50.T @ b_perturbed_noise_small_50)
x_noise_large_50 = np.linalg.solve(A_50.T @ A_50, A_50.T @ b_perturbed_noise_large_50)
x_noise_small_25 = np.linalg.solve(A_25.T @ A_25, A_25.T @ b_perturbed_noise_small_25)
x_noise_large_25 = np.linalg.solve(A_25.T @ A_25, A_25.T @ b_perturbed_noise_large_25)


x_new_noise_small_100 = np.reshape(x_noise_small_100, np.shape(downsized_im_100)) - np.min(x_noise_small_100)
x_new_noise_large_100 = np.reshape(x_noise_large_100, np.shape(downsized_im_100)) - np.min(x_noise_large_100)
x_new_noise_small_50 = np.reshape(x_noise_small_50, np.shape(downsized_im_50)) - np.min(x_noise_small_50)
x_new_noise_large_50 = np.reshape(x_noise_large_50, np.shape(downsized_im_50)) - np.min(x_noise_large_50)
x_new_noise_small_25 = np.reshape(x_noise_small_25, np.shape(downsized_im_25)) - np.min(x_noise_small_25)
x_new_noise_large_25 = np.reshape(x_noise_large_25, np.shape(downsized_im_25)) - np.min(x_noise_large_25)


# Figure

fig, ax = plt.subplots(2,3)
im1 = ax[0,1].imshow(np.sqrt(x_new_noise_small_50), cmap='viridis')
ax[0,1].set_title('Noise small_50')
fig.colorbar(im1, ax=ax[0,1])

im2 = ax[0,0].imshow(np.sqrt(x_new_noise_small_100), cmap='viridis')
ax[0,0].set_title('Noise small_100')
fig.colorbar(im2, ax=ax[0,0])

im3 = ax[1,0].imshow(np.sqrt(x_new_noise_large_100), cmap='viridis')
ax[1,0].set_title('Noise large_100')
fig.colorbar(im3, ax=ax[1,0])

im3 = ax[1,1].imshow(np.sqrt(x_new_noise_large_50), cmap='viridis')
ax[1,1].set_title('Noise large_50')
fig.colorbar(im3, ax=ax[1,1])

im1 = ax[0,2].imshow(np.sqrt(x_new_noise_small_25), cmap='viridis')
ax[0,2].set_title('Noise small_25')
fig.colorbar(im1, ax=ax[0,1])

im1 = ax[1,2].imshow(np.sqrt(x_new_noise_small_25), cmap='viridis')
ax[1,2].set_title('Noise small_25')
fig.colorbar(im1, ax=ax[0,1])

plt.show()


