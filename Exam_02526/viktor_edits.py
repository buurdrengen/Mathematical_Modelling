import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import paralleltomo # [A,theta,p,d] = paralleltomo(N,theta,p,d)


# Load the image with lead and steel shot
data = np.array(scipy.io.loadmat("Exam_02526/testImage.mat")['im']) #Pixel size: 0.1 mm
downsized_im = data[::100, ::100] # pixel size: 10 mm
x = np.copy(downsized_im.flatten())

# Initialize system matrix and stuff
N = downsized_im.shape[0]
theta = np.array([np.arange(0, 180, 1)])
p = 50
d = 0.5 # [m]
[A,theta,p,d] = paralleltomo.paralleltomo(N, theta, p)


# Load simulated forward projection 
b = A @ x # Finding b
b_perturbed = b + np.random.normal(0, 0.1, size=np.shape(b)) # adding noise

print(np.shape(A))
print(np.shape(x))
print(np.shape(b))

x_new = np.linalg.solve(A.T @ A, A.T @ b_perturbed)

print(np.sum(x_new-x)) 


# plt.hist(b)
# plt.imshow(data)
plt.show()









