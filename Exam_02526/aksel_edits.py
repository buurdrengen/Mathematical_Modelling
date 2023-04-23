import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import paralleltomo # [A,theta,p,d] = paralleltomo(N,theta,p,d)


# Load the image with lead and steel shot
data = np.array(scipy.io.loadmat("Exam_02526/testImage.mat")['im']) #Pixel size: 0.1 mm
downsized_im = data[::100, ::100] # pixel size: 10 mm
x = np.copy(downsized_im.flatten())

# Find the different attenuation constants
[x_const, x_const_count] = np.unique(x, return_counts=True)
print(f"the attenuation constanstants and the amount of times they are \
in the im is: \nconstants: {x_const}\ncounts:    {x_const_count}")

# Define the perbuations
noise = 1e-3
theta = np.array([np.arange(0, 180, 1)])
p = 50
d = 0.5 # [m]

# Initialize system matrix and stuff
N = downsized_im.shape[0]
[A,theta,p,d] = paralleltomo.paralleltomo(N, theta, p)


# Load simulated forward projection 
b = A @ x # Finding b
b_perturbed = b + np.random.normal(0, noise, size=np.shape(b)) # adding noise

x_new = np.linalg.solve(A.T @ A, A.T @ b_perturbed)


print(np.max(x_new-x))

x_new_pic = np.reshape(x_new, np.shape(downsized_im)) - np.min(x_new)

print(np.shape(x))
print(np.shape(A))
print(np.shape(x_new))
print(np.shape(x_new_pic))


fig, ax = plt.subplots(ncols=2, figsize=(10,6))
im1 = ax[0].imshow(np.sqrt(downsized_im), cmap='viridis')
fig.colorbar(im1, ax=ax[0])
im2 = ax[1].imshow(np.sqrt(x_new_pic), cmap='viridis')
fig.colorbar(im2, ax=ax[1])
plt.show()



