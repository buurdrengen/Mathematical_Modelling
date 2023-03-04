import numpy as np
import matplotlib.pyplot as plt
import os 
import helpFunctions as hf
from compare_plot import *
from spectral_analysis import *

# for i in range(5):
imName = "Salami/multispectral_day01.mat"
annotationName = "Salami/annotation_day01.png"
[multiIm, annotationIm] = hf.loadMulti(imName, annotationName)

fat_vector = multiIm[annotationIm[:,:,1], :]
meat_vector = multiIm[annotationIm[:,:,2], :]


print(np.shape(annotationIm[:,:, 0]))
print(np.shape(multiIm[:,:]))
print(np.shape(fat_vector))

# amount of points in each annoted category for day 1
m_fat = np.size(fat_vector[:,0])
m_meat = np.size(meat_vector[:,0])

# Means and standard devations for fat and meat categories
fat_vector_means = np.mean(fat_vector, axis=0)
meat_vector_means = np.mean(meat_vector, axis=0)
fat_vector_sd = np.std(fat_vector, axis=0)
meat_vector_sd = np.std(meat_vector, axis=0)

# Finding the combined standard devation for fat and meat as
combined_sd = 1/(m_fat + m_meat) * (m_fat*fat_vector_sd + m_meat*meat_vector_sd) # (strictly speaking useless)

# finding the simple threshold under the assumption that the standard devs are the same corresponds
# to finding the mid point between the 2 means
t = (fat_vector_means + meat_vector_means)/2

# Finding the error classification rates
error_rate = ( np.sum(fat_vector < t, axis=0) + np.sum(meat_vector > t, axis=0)  ) / (m_fat + m_meat)
print(np.round(error_rate*100,2))

# The best spectral band for this simple analysis is
best_band = np.argmin(error_rate) # This is 0-indexed
print(f"the best band is { best_band+1}")

# the background index is the not index of all the annotations added together
index_background = (annotationIm[:,:,0]+annotationIm[:,:,1]+annotationIm[:,:,2])==0

# Classifying the entire image with band 14 gives us
entire_im = np.copy(multiIm[:,:,best_band])
index_fat = (entire_im > t[best_band])
index_meat = (entire_im < t[best_band])
entire_im[index_fat] = 1 #set all fat values to class 1
entire_im[index_meat] = 2 # set all meat values to class 2
entire_im[index_background] = 0 # set background to class 0

compare_image(image = entire_im, day = 1 , title = "Day 1 - Classification by Threshold Value")

#plt.imshow(entire_im)


"""
Figures 
"""

# plt.figure()
# for j in range(19):
#     plt.clf()
#     fat_hist = plt.hist(fat_vector[:, j], range=[0, 100], bins=50, color='orange')
#     meat_hist = plt.hist(meat_vector[:, j], range=[0,100], bins=50, color='blue')
#     plt.title(f"band {j}")
#     # plt.legend('fat', 'meat')
#     plt.pause(1/2)

compare_spectrum(fat_vector_means, fat_vector_sd, meat_vector_means, meat_vector_sd, title = "Mean Intensity of Sausage at Day 1")



##  Discriminant method
Sigma_fat = np.cov(fat_vector, rowvar = False)
Sigma_meat = np.cov(meat_vector, rowvar = False)

m_fat = np.size(fat_vector,0) - 1
m_meat = np.size(meat_vector,0) - 1

Sigma = 1/(m_fat + m_meat) * (m_fat*Sigma_fat + m_meat * Sigma_meat)

Sigma_inv = np.linalg.inv(Sigma)

mu_fat = np.copy(fat_vector_means)
mu_meat = meat_vector_means

XT = np.reshape(multiIm,(514*514,19))

S_fat = (np.linalg.multi_dot([XT, Sigma_inv, mu_fat]) - 1/2 * np.linalg.multi_dot([mu_fat.T, Sigma_inv, mu_fat])).reshape((514,514))
S_meat = (np.linalg.multi_dot([XT, Sigma_inv, mu_meat]) - 1/2 * np.linalg.multi_dot([mu_meat.T, Sigma_inv, mu_meat])).reshape((514,514))

# Post processing

true_fat_index = annotationIm[:,:,1]
true_meat_index = annotationIm[:,:,2]

index_fat2 = (S_fat > S_meat)
index_meat2 = (S_meat >= S_fat)
entire_im2 = construct_entire_im(index_fat2,index_meat2,index_background)

#Chekc the result
compute_errorrate(true_fat_index, true_meat_index, index_fat2, index_meat2)
compare_image(image = entire_im2, day = 1 , title = "Day 1 - Classification by Multivariate Linear Discriminant")