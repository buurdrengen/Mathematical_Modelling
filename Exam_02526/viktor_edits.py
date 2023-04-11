import numpy as np
import scipy.io
import paralleltomo # [A,theta,p,d] = paralleltomo(N,theta,p,d)


# Load the image with lead and steel shot
data = np.array(scipy.io.loadmat("Exam_02526/testImage.mat")['im']) #Pixel size: 0.1 mm
downsized_im = data[::100, ::100] # pixel size: 10 mm


# Initialize settings for paralleltomo
N = downsized_im.shape[0]
theta = np.array([np.arange(0, 180, 1)])
theta =np.matrix(np.linspace(0,179,179))
print(theta)
quit()
[A,theta,p,d] = paralleltomo.paralleltomo(N, theta, 50) 

# Load system matrix A
paralleltomo







