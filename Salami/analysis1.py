import numpy as np
import matplotlib.pyplot as plt
import os 
import helpFunctions as hf
import spectral_analysis as sa
import compare_plot as MW_plot

# for i in range(5):
imName = "Salami/multispectral_day01.mat"
annotationName = "Salami/annotation_day01.png"
[multiIm, annotationIm] = hf.loadMulti(imName, annotationName)

fat_vector = multiIm[annotationIm[:,:,1], :]
meat_vector = multiIm[annotationIm[:,:,2], :]


# print(np.shape(annotationIm[:,:, 0]))
# print(np.shape(multiIm[:,:]))
# print(np.shape(fat_vector))

# amount of points in each annoted category for day 1
m_fat = np.size(fat_vector[:,0])
m_meat = np.size(meat_vector[:,0])

"""
Threshold value for a single spectral band
"""

# Means and standard devations for fat and meat categories
fat_vector_means = np.mean(fat_vector, axis=0)
meat_vector_means = np.mean(meat_vector, axis=0)
fat_vector_sd = np.std(fat_vector, axis=0)
meat_vector_sd = np.std(meat_vector, axis=0)

# Finding the combined standard devation for fat and meat as
# combined_sd = 1/(m_fat + m_meat) * (m_fat*fat_vector_sd + m_meat*meat_vector_sd) # (strictly speaking useless)

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

MW_plot.compare_image(entire_im)


"""
Classification by means of all spectral bands
"""
# Loading the variables to be used in analysis
[multiIm, annotationIm] = hf.loadMulti(imName, annotationName)
fat_vector = multiIm[annotationIm[:,:,1], :]
meat_vector = multiIm[annotationIm[:,:,2], :]

# Finding the mean vector (again))
fat_vector_means = np.mean(fat_vector, axis=0)
meat_vector_means = np.mean(meat_vector, axis=0)

# covariance matrix for all spectral bands (Sigma_sb = sigma spectral bands)
Sigma_fat = np.cov(fat_vector.T)
Sigma_meat = np.cov(meat_vector.T)
Sigma = 1/(m_fat + m_meat - 2) * ((m_fat-1)*Sigma_fat + (m_meat-1) * Sigma_meat)
Sigma_inv = np.linalg.inv(Sigma)
# Finding the annoted examples
spectral_fat = multiIm[annotationIm[:,:,1]]
spectral_meat = multiIm[annotationIm[:,:,2]]

# we assume no prior probability
S_fat_fat = spectral_fat.dot(Sigma_inv).dot(fat_vector_means) - 1/2 * fat_vector_means.T.dot(Sigma_inv).dot(fat_vector_means)
S_meat_fat = spectral_fat.dot(Sigma_inv).dot(meat_vector_means) - 1/2 * meat_vector_means.T.dot(Sigma_inv).dot(meat_vector_means)
S_fat_meat = spectral_meat.dot(Sigma_inv).dot(fat_vector_means) - 1/2 * fat_vector_means.T.dot(Sigma_inv).dot(fat_vector_means)
S_meat_meat = spectral_meat.dot(Sigma_inv).dot(meat_vector_means) - 1/2 * meat_vector_means.T.dot(Sigma_inv).dot(meat_vector_means)




false_fat = np.sum(S_meat_fat >= S_fat_fat) # annotationIm[:,:,1] is annotated fat
false_meat = np.sum(S_fat_meat > S_meat_meat) # annotationIm[:,:,2] is annotated meat

# calculate the errors
fat_error = false_fat/m_fat
meat_error = false_meat/m_meat

print(f"the classification error rate is {np.round(fat_error*100,1)} % for fat and {np.round(meat_error*100,1)} % for meat")

index_background = (annotationIm[:,:,0]+annotationIm[:,:,1]+annotationIm[:,:,2])==0
index_fat = annotationIm[:,:, 1]
index_meat = annotationIm[:,:,2]

# the background index is the not index of all the annotations added together
index_background = (annotationIm[:,:,0]+annotationIm[:,:,1]+annotationIm[:,:,2])==0

entire_im = np.copy(multiIm[:,:,:]) # Copying the image to be classified

# Defining the S_labels for the entire image.
S_fat = entire_im.dot(Sigma_inv).dot(fat_vector_means) - 1/2 * fat_vector_means.T.dot(Sigma_inv).dot(fat_vector_means)
S_meat = entire_im.dot(Sigma_inv).dot(meat_vector_means) - 1/2 * meat_vector_means.T.dot(Sigma_inv).dot(meat_vector_means)


index_fat = (S_fat >= S_meat)
index_meat = (S_meat > S_fat)
entire_im[index_fat, 0] = 1 #set all fat values to class 1
entire_im[index_meat, 0] = 2 # set all meat values to class 2
entire_im[index_background, 0] = 0 # set background to class 0

# MW_plot.compare_image(entire_im[:,:, 0])
# plt.show()



"""
error rates for all days
"""
day_list = ['01', '06', '13', '20', '28']

# Create a matrix for storing all error rates in
error_meat = np.zeros((5,5))
error_fat = np.zeros((5,5))
# Fill all values with nan
error_meat[:] = np.nan
error_fat[:] = np.nan

for i, model_day in enumerate(day_list):
    # Finding the spectral images and annoteted indices for 
    imName = f"Salami/multispectral_day{model_day}.mat"
    annotationName = f"Salami/annotation_day{model_day}.png"
    [multiIm, annotationIm] = hf.loadMulti(imName, annotationName)

    # Define the annoted fat and meat vectors
    fat_vector = multiIm[annotationIm[:,:,1], :]
    meat_vector = multiIm[annotationIm[:,:,2], :]
    
    # Finding the mean vectors for the classes wrt. all variables. 
    fat_vector_means = np.mean(fat_vector, axis=0)
    meat_vector_means = np.mean(meat_vector, axis=0)

    # covariance matrix for all spectral bands
    Sigma_fat = np.cov(fat_vector.T)
    Sigma_meat = np.cov(meat_vector.T)
    Sigma = 1/(m_fat + m_meat - 2) * ((m_fat-1)*Sigma_fat + (m_meat-1) * Sigma_meat)
    Sigma_inv = np.linalg.inv(Sigma)


    for j, day in enumerate(day_list):
        if day == model_day:
            continue
        # Load data
        imName = f"Salami/multispectral_day{day}.mat"
        annotationName = f"Salami/annotation_day{day}.png"
        [multiIm, annotationIm] = hf.loadMulti(imName, annotationName)

        # Finding the annoted examples
        annetoted_fat = multiIm[annotationIm[:,:,1]]
        spectral_meat = multiIm[annotationIm[:,:,2]]

        # S(x) in the fat regions
        S_fat = annetoted_fat.dot(Sigma_inv).dot(fat_vector_means) - 1/2 * fat_vector_means.T.dot(Sigma_inv).dot(fat_vector_means)
        S_meat = annetoted_fat.dot(Sigma_inv).dot(meat_vector_means) - 1/2 * meat_vector_means.T.dot(Sigma_inv).dot(meat_vector_means)
        false_fat = np.sum(S_meat >= S_fat) 

        # S(x) in the meat regions
        S_fat = spectral_meat.dot(Sigma_inv).dot(fat_vector_means) - 1/2 * fat_vector_means.T.dot(Sigma_inv).dot(fat_vector_means)
        S_meat = spectral_meat.dot(Sigma_inv).dot(meat_vector_means) - 1/2 * meat_vector_means.T.dot(Sigma_inv).dot(meat_vector_means)
        false_meat = np.sum(S_fat > S_meat)

        # calculate the errors
        error_fat[i,j] = false_fat/m_fat
        error_meat[i,j] = false_meat/m_meat

        print(f"when the model is trained on day {model_day} and tested on day {day} the classification error rate is {np.round(error_fat[i,j]*100,2)} % for fat and {np.round(error_meat[i,j]*100,2)} % for meat")
    print("\n")

# Generate a color ramp to be used for plotting
color_ramp = ['#FF1010', '#C51235', '#8C145A', '#52167F', '#1919A4']

fig, (ax1, ax2) = plt.subplots(2, sharex=True, constrained_layout=True)
for i in range(5):
    # ax.plot(day_list, error_meat[i]*100, label= f"model from day {day_list[i]}", color = color_ramp[i], marker = '.', linestyle = '', markersize = 15)
    ax1.bar(np.linspace(10-i, 50-i, 5), error_meat[i]*100, width = 1, color = color_ramp[i], label=f"model from day {day_list[i]}")
    ax2.bar(np.linspace(10-i, 50-i, 5), error_fat[i]*100, width = 1, color = color_ramp[i])
    
ax1.set_ylabel('meat error rate [%]')
ax2.set_ylabel('fat error rate [%]')
ax2.set_xlabel('day')
ax1.set_xticks(np.linspace(10-3, 50-3, 5), day_list)
fig.legend()
plt.savefig('Salami/error_rates.png')



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


# plt.figure()
# plt.plot(np.mean(fat_vector, axis=0), label = 'fat', color='orange')
# plt.plot(np.mean(meat_vector, axis=0), label='meat', color='blue')
# plt.legend()
# plt.xticks(range(19))
# plt.grid()
# plt.show()


for day in day_list:
    imName = f"Salami/multispectral_day{day}.mat"
    annotationName = f"Salami/annotation_day{day}.png"
    [multiIm, annotationIm] = hf.loadMulti(imName, annotationName)

    # Define the annoted fat and meat vectors
    fat_vector = multiIm[annotationIm[:,:,1], :]
    meat_vector = multiIm[annotationIm[:,:,2], :]

    # Finding the mean vectors for the classes wrt. all variables. 
    fat_vector_means = np.mean(fat_vector, axis=0)
    meat_vector_means = np.mean(meat_vector, axis=0)

    # Finding the standard devations wrt all variables
    fat_vector_sd = np.std(fat_vector, axis=0)
    meat_vector_sd = np.std(meat_vector, axis=0)

    
    spectrum = np.array([410, 438, 450, 468, 502, 519, 572, 591, 625, 639, 653, 695, 835, 863, 880, 913, 929, 940, 955])
    plt.figure()
    plt.errorbar(spectrum, fat_vector_means, 2*fat_vector_sd, capsize=5, ecolor='black', label='fat', color='orange')
    plt.errorbar(spectrum, meat_vector_means, 2*meat_vector_sd, capsize=5, ecolor='black',  label='meat', color='blue')
    plt.legend()
    plt.grid()
    plt.xlabel("Frequency [nm]")
    plt.ylabel("Mean Intensity [pixel value]")
    plt.title(f"Mean Intensity of Sausage at Day {day}")
    plt.savefig(f"Salami/variance_plot_day{day}.png")
    # plt.show()