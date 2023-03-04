import numpy as np
import matplotlib.pyplot as plt
import os 
import helpFunctions as hf







def load_day_data(day):
    str_day = "0"*(day<10)+str(day)
    imName = "Salami/multispectral_day" + str_day + ".mat"
    annotationName = "Salami/annotation_day" + str_day + ".png"
    [multiIm, annotationIm] = hf.loadMulti(imName, annotationName)
    
    true_fat = annotationIm[:,:,1]
    true_meat = annotationIm[:,:,2]
    index_background = (annotationIm[:,:,0] + annotationIm[:,:,1] + annotationIm[:,:,2]) == 0

    fat_vector = multiIm[true_fat, :]
    meat_vector = multiIm[true_meat, :]

    return multiIm, true_fat, true_meat, index_background, fat_vector, meat_vector

# ---------------------------------------------------------------------

def compute_errorrate(true_fat, true_meat, index_fat, index_meat, method_name = None):

    errorrate = (np.sum(index_fat[true_fat] == 0) + np.sum(index_meat[true_meat] == 0)) / (np.sum(true_fat) + np.sum(true_meat))

    if method_name != None: 
        error_fat = np.sum(index_fat[true_fat] == 0)/np.sum(true_fat)
        error_meat = np.sum(index_meat[true_meat] == 0)/np.sum(true_meat)
        
        print(f"{method_name} has an error rate of {error_fat*100:.2f}% for fat and {error_meat*100:.2f}% for meat for a total of {errorrate*100:0.2f}%")
    
    return errorrate

# ---------------------------------------------------------------------

def construct_entire_im(index_fat, index_meat, index_background):
    entire_im = np.zeros(np.shape(index_fat))
    entire_im[index_fat] = 1
    entire_im[index_meat] = 2
    entire_im[index_background] = 0
    return entire_im
# ---------------------------------------------------------------------

def train_multivariate_linear_discriminant(multiIm, fat_vector, meat_vector):

    # Preprocessing
    m_fat = np.size(fat_vector,0) - 1
    m_meat = np.size(meat_vector,0) - 1

    mean_fat = np.mean(fat_vector,axis=0)
    mean_meat = np.mean(meat_vector,axis=0)

    k = [np.size(multiIm,0), np.size(multiIm,1)]

    # Processing
    Sigma_fat = np.cov(fat_vector, rowvar = False)
    Sigma_meat = np.cov(meat_vector, rowvar = False)

    Sigma = 1/(m_fat + m_meat) * (m_fat*Sigma_fat + m_meat * Sigma_meat)

    Sigma_inv = np.linalg.inv(Sigma)

    mu_fat = np.copy(mean_fat)
    mu_meat = mean_meat

    return Sigma_inv, mu_fat, mu_meat

# ---------------------------------------------------------------------

def compute_multivariate_linear_discriminant(multiIm, Sigma_inv, mu_fat, mu_meat, pi = 1):

    k = [np.size(multiIm,0), np.size(multiIm,1)]
    XT = np.reshape(multiIm,(k[0]*k[1],np.size(mu_fat)))

    S_fat = (np.linalg.multi_dot([XT, Sigma_inv, mu_fat]) - 1/2 * np.linalg.multi_dot([mu_fat.T, Sigma_inv, mu_fat]) + np.log(pi)).reshape(k)
    S_meat = (np.linalg.multi_dot([XT, Sigma_inv, mu_meat]) - 1/2 * np.linalg.multi_dot([mu_meat.T, Sigma_inv, mu_meat]) + np.log(pi)).reshape(k)
    
    index_fat = (S_fat > S_meat)
    index_meat = (S_fat <= S_meat)

    return index_fat, index_meat

# ---------------------------------------------------------------------

def train_threshold_value(fat_vector, meat_vector):

    # Preprocessing
    m_fat = np.size(fat_vector,0)
    m_meat = np.size(meat_vector,0)
    mean_fat = np.mean(fat_vector,axis=0)
    mean_meat = np.mean(meat_vector,axis=0)

    t = (mean_fat + mean_meat)/2

    error_rate = ( np.sum(fat_vector < t, axis=0) + np.sum(meat_vector > t, axis=0)  ) / (m_fat + m_meat)
    best_band = np.argmin(error_rate)

    return t, best_band

# ---------------------------------------------------------------------

def compute_threshold_value(multiIm, t, best_band):
    best_band_im = multiIm[:,:,best_band]
    index_fat = (best_band_im > t[best_band])
    index_meat = (best_band_im <= t[best_band])

    return index_fat, index_meat


if __name__ == "__main__":
    multiIm, true_fat, true_meat, index_background, fat_vector, meat_vector = load_day_data(1)

    # Threshold Value
    t, best_band = train_threshold_value(fat_vector, meat_vector)
    index_fat, index_meat = compute_threshold_value(multiIm,t, best_band)
    errorrate_1 = compute_errorrate(true_fat, true_meat, index_fat, index_meat, method_name = "TV")

    # Multivariate Linear Discriminant
    Sigma_inv, mu_fat, mu_meat = train_multivariate_linear_discriminant(multiIm, fat_vector, meat_vector)
    index_fat, index_meat = compute_multivariate_linear_discriminant(multiIm, Sigma_inv, mu_fat, mu_meat)
    errorrate_2 = compute_errorrate(true_fat, true_meat, index_fat, index_meat, method_name = "MLD")

    #Evaluation
    print("The best method is " + "TV"*int(errorrate_1 < errorrate_2) + "MLD"*int(errorrate_1 > errorrate_2))
        