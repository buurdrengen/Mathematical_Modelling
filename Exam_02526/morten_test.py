import numpy as np
from phanton_generator import phantom
from paralleltomo import paralleltomo
from scipy.linalg import solve
from skimage.measure import block_reduce

data = phantom(2, 2)

res = 100
p = 225
angle_no = 180

base_array = np.zeros(np.shape(data))   
known_wood = np.copy(base_array); known_wood[(data > 0) & (data < 6.5e-1)] = 1
known_iron= np.copy(base_array); known_iron[data == 6.5e-1] = 1
known_lead = np.copy(base_array); known_lead[data == 3.43] = 1

# Downsample image
downsizing = 5e3//res # define downsizing constant
downsized_im = block_reduce(data, block_size=(downsizing, downsizing), func=np.mean) # downsample the image
x = np.copy(downsized_im.flatten()) # flatten the attenuation constants

downsized_known_wood = block_reduce(known_wood, block_size=downsizing, func=np.min).astype(int) # downsample the image
downsized_known_iron = block_reduce(known_iron, block_size=downsizing, func=np.max).astype(int) # downsample the image
downsized_known_lead = block_reduce(known_lead, block_size=downsizing, func=np.max).astype(int) # downsample the image

# Find the separators based on the anotted masks
x_imShape = x.reshape(np.shape(downsized_im)) 
# Define the attenuation coefficient means and find the separators betweeen classes

# number of each cell
N_lead = np.sum(downsized_known_lead)
N_iron = np.sum(downsized_known_iron)
N_wood = np.sum(downsized_known_wood)

theta = np.array([np.linspace(0, 180, angle_no)]) 
d = 0.5 # [m]
N = downsized_im.shape[0]
[A,theta,p,d] = paralleltomo(N, theta, p)

