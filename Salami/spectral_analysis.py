import numpy as np
import helpFunctions as hf
from compare_plot import compare_train


def load_day_data(day):
    """ Loads relevant data for a given day for futher computation

    Args:
        day (int): The day for which to search for data. Will give an error if no files are found.

    Returns:
        multiIm (ndarray): Gray-scale values for all pixels in the image across all bands
        true_fat (ndarray): Index of all pixels known to be fat
        true_meat (ndarray): Index of all pixels known to be meat
        index_background (ndarray): Index of all pixels known to be background
        fat_vector (ndarray): Gray-scale values of all pixels known to be fat
        meat_vector (ndarray): Gray-scale values of all pixels known to be meat
    """
    
    str_day = "0"*int(day<10)+str(day)
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
    """ Calculates the error rate given a guess and known indecies

    Args:
        true_fat (ndarray): Index of all pixels known to be fat
        true_meat (ndarray): Index of all pixels known to be meat
        index_fat (ndarray): Index of pixels guessed to be fat
        index_meat (ndarray): Index of pixels guessed to be meat
        method_name (str, optional): If given, this will display additional information about the guess. Defaults to None.

    Returns:
        error_rate (float): The fraction of guessed pixels known to be wrong over the total number of known pixels.
    """

    error_rate = (np.sum(index_fat[true_fat] == 0) + np.sum(index_meat[true_meat] == 0)) / (np.sum(true_fat) + np.sum(true_meat))

    if method_name: 
        error_fat = np.sum(index_fat[true_fat] == 0)/np.sum(true_fat)
        error_meat = np.sum(index_meat[true_meat] == 0)/np.sum(true_meat)
        
        print(f"{method_name} has an error rate of {error_fat*100:.2f}% for fat and {error_meat*100:.2f}% for meat for a total of {error_rate*100:0.2f}%")
    
    return error_rate

# ---------------------------------------------------------------------

def construct_entire_im(index_fat, index_meat, index_background, cmap = None):
    """ Constructs a plottable image of a given image guess guess from their indicies. 
        Any pixel not covered in any input will be assumed to be background

    Args:
        index_fat (ndarray): Index of pixels guessed to be fat
        index_meat (ndarray): Index of pixels guessed to be meat
        index_background (ndarray): Index of all pixels known to be background
        cmap (str, optional): If given, this will try to correct the image values for a colormap. Defaults to None.

    Returns:
        entire_im (ndarray): A plottable image of the given guess
    """
    
    entire_im = np.zeros(np.shape(index_fat))
    entire_im[index_fat] = 1 -2*int(cmap == "coolwarm") + 1*int(cmap in ["hot", "gray"])
    entire_im[index_meat] = 2 - 1*int(cmap in ["coolwarm", "gray"]) -3*int(cmap=="BrBG") - 1.6*int(cmap == "hot")
    entire_im[index_background] = 0 + 3*int(cmap == "RdPu") + 1.5*int(cmap == "PuRd")
    return entire_im
# ---------------------------------------------------------------------

def train_multivariate_linear_discriminant(fat_vector, meat_vector):
    """ Trains a model of a multivariate linear discriminant with known values of fat and meat

    Args:
        fat_vector (ndarray): Gray-scale values of all pixels known to be fat
        meat_vector (ndarray): Gray-scale values of all pixels known to be meat

    Returns:
        Sigma_inv (ndarray): The inverse Sigma covariance matrix
        mu_fat (ndarray): The mean of known fat values
        mu_meat (ndarray) The mean of known meat values
    """

    # Preprocessing
    m_fat = np.size(fat_vector,0) - 1
    m_meat = np.size(meat_vector,0) - 1

    mu_fat = np.mean(fat_vector,axis=0)
    mu_meat = np.mean(meat_vector,axis=0)


    # Processing
    Sigma_fat = np.cov(fat_vector, rowvar = False)
    Sigma_meat = np.cov(meat_vector, rowvar = False)

    Sigma = 1/(m_fat + m_meat) * (m_fat*Sigma_fat + m_meat * Sigma_meat)

    Sigma_inv = np.linalg.inv(Sigma)

    return Sigma_inv, mu_fat, mu_meat

# ---------------------------------------------------------------------

def compute_multivariate_linear_discriminant(multiIm, Sigma_inv, mu_fat, mu_meat, pi = 0.5):
    """ Computes and evaluates the multivariate linear discriminant for the given image.

    Args:
        multiIm (ndarray): Gray-scale values for all pixels in the image across all bands
        Sigma_inv (ndarray): The inverse Sigma covariance matrix
        mu_fat (ndarray): The mean of known fat values
        mu_meat (ndarray) The mean of known meat values
        pi (float, optional): Prior probability that any given pixel is fat. Defaults to 0.5.

    Returns:
        index_fat (ndarray): Index of pixels guessed to be fat
        index_meat(ndarray): Index of pixels guessed to be meat
    """

    k = [np.size(multiIm,0), np.size(multiIm,1)]
    XT = np.reshape(multiIm,(k[0]*k[1],np.size(mu_fat)))

    S_fat = (np.linalg.multi_dot([XT, Sigma_inv, mu_fat]) - 1/2 * np.linalg.multi_dot([mu_fat.T, Sigma_inv, mu_fat]) + np.log(pi)).reshape(k)
    S_meat = (np.linalg.multi_dot([XT, Sigma_inv, mu_meat]) - 1/2 * np.linalg.multi_dot([mu_meat.T, Sigma_inv, mu_meat]) + np.log(1-pi)).reshape(k)
    
    index_fat = (S_fat > S_meat)
    index_meat = (S_fat <= S_meat)

    return index_fat, index_meat

# ---------------------------------------------------------------------

def train_threshold_value(fat_vector, meat_vector):
    """ Computes the threshold values with known values of fat and meat

    Args:
        fat_vector (ndarray): Gray-scale values of all pixels known to be fat
        meat_vector (ndarray): Gray-scale values of all pixels known to be meat

    Returns:
        t (ndarray): The threshold for all bands
        best_band (int): Index of the band with lowest error rate
    """

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
    """ Evaluates the threshold values for the given image.

    Args:
        multiIm (ndarray): Gray-scale values for all pixels in the image across all bands
        t (ndarray): The threshold for all bands
        best_band (int): Index of the band with lowest error rate

    Returns:
        index_fat (ndarray): Index of pixels guessed to be fat
        index_meat(ndarray): Index of pixels guessed to be meat
    """
    best_band_im = multiIm[:,:,best_band]
    index_fat = (best_band_im > t[best_band])
    index_meat = (best_band_im <= t[best_band])

    return index_fat, index_meat

# ---------------------------------------------------------------------

def alpha(train_day, compare_days, cmap = None, save_fig = False, show_fig = True, print_error = True, pi = 0.5):
    """ Performs cross-comparisons of a training set and a set of given days.
    
    Please find a better name for this function...

    Args:
        train_day (int): The day for which the models will get their training data.
        compare_days (ndarray): The days to cross-compare with the trained models.
        cmap (str, optional): If given, this will apply a specific colormap to the plot. Defaults to None.
        save_fig (bool, optional): If true, this will save the plot to a file on disk. Defaults to False.
        show_fig (bool, optional): If true, this will display the plot. Defaults to True.
        print_error (bool, optional): If true, this will print information about the error rate. Defaults to True.
        pi (float, optional): Prior probability that any given pixel is fat. Defaults to 0.5.

    Returns:
        err (ndarray): Array of computed error rates
    """ 
    
    if np.shape(compare_days) == (): compare_days = np.array([compare_days])
    save_name = None
    
    #Training
    multiIm, _, _, _, fat_vector, meat_vector = load_day_data(train_day)
    t, best_band = train_threshold_value(fat_vector, meat_vector)
    Sigma_inv, mu_fat, mu_meat = train_multivariate_linear_discriminant(fat_vector, meat_vector)
    
    err = np.zeros((np.size(compare_days),2))
    
    # Cross-comparison
    for j,i in enumerate(compare_days):
        if print_error: print("\nDay " + str(i))
        multiIm, true_fat, true_meat, index_background, _, _ = load_day_data(i)
        index_fat_tv, index_meat_tv = compute_threshold_value(multiIm,t, best_band)
        errorrate_tv = compute_errorrate(true_fat, true_meat, index_fat_tv, index_meat_tv, method_name = "TV"*int(print_error))
        index_fat_mld, index_meat_mld = compute_multivariate_linear_discriminant(multiIm, Sigma_inv, mu_fat, mu_meat, pi = pi)
        errorrate_mld = compute_errorrate(true_fat, true_meat, index_fat_mld, index_meat_mld, method_name = "MLD"*int(print_error))
        
        err[j,:] = [errorrate_tv,errorrate_mld]
        
        # Evaluation
        if print_error: print("The best method is " + "TV"*int(errorrate_tv < errorrate_mld) + "MLD"*int(errorrate_tv > errorrate_mld))
        
        # Plot
        if save_fig: save_name = f"day{i}_t{train_day}"
        imagetv = construct_entire_im(index_fat_tv, index_meat_tv, index_background, cmap=cmap)
        imagemld = construct_entire_im(index_fat_mld, index_meat_mld, index_background, cmap=cmap)
        compare_train(imagetv, imagemld, day = i, title = f"Day {i} Trained on Day {train_day}", cmap=cmap, save_fig = save_name, show_fig = show_fig)
        
    
    return err


# ---------------------------------------------------------------------

if __name__ == "__main__":

    # cmap: implemented:
    #   coolwarm
    #   BrBGm
    #   GnBu
    #   hot
    #   RdPu
    #   PuRd
    #   gray
        
        
    liste = [1,6,13,20,28]
    result = np.zeros((np.size(liste), np.size(liste), 2))
    for j,i in enumerate(liste):
        print("\n" + 25*"-")
        print(f"Training on day {i}")
        result[j,:,:] = alpha(i,liste, cmap = "seismic", save_fig=False, show_fig = False, print_error = False)
    
    print("Data for TV:")
    print(np.round(result[:,:,0]*100,2))
    print(25*"-")
    print("Data for MLD:")
    print(np.round(result[:,:,1]*100,2))
    
    I = np.eye(np.size(liste), dtype=int)
    
    result[:,:,0] = result[:,:,0] -I*result[:,:,0]
    result[:,:,1] = result[:,:,1] -I*result[:,:,1]
    
    # print(result[:,:,0])
    print(25*"-")
    print("Mean Values:")
    print(np.round(np.sum(result*25,axis=1),2))