import numpy as np
import helpFunctions as hf
from compare_plot import compare_train


def load_day_data(day):
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

    errorrate = (np.sum(index_fat[true_fat] == 0) + np.sum(index_meat[true_meat] == 0)) / (np.sum(true_fat) + np.sum(true_meat))

    if method_name: 
        error_fat = np.sum(index_fat[true_fat] == 0)/np.sum(true_fat)
        error_meat = np.sum(index_meat[true_meat] == 0)/np.sum(true_meat)
        
        print(f"{method_name} has an error rate of {error_fat*100:.2f}% for fat and {error_meat*100:.2f}% for meat for a total of {errorrate*100:0.2f}%")
    
    return errorrate

# ---------------------------------------------------------------------

def construct_entire_im(index_fat, index_meat, index_background, cmap = None):
    entire_im = np.zeros(np.shape(index_fat))
    entire_im[index_fat] = 1 -2*int(cmap == "coolwarm") + 1*int(cmap in ["hot", "gray"])
    entire_im[index_meat] = 2 - 1*int(cmap in ["coolwarm", "gray"]) -3*int(cmap=="BrBG") - 1.6*int(cmap == "hot")
    entire_im[index_background] = 0 + 3*int(cmap == "RdPu") + 1.5*int(cmap == "PuRd")
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

# ---------------------------------------------------------------------

def alpha(train_day, compare_days, plot = False, cmap = None, save_fig = False, show_fig = True, print_error = True):
    
    if np.shape(compare_days) == (): compare_days = np.array([compare_days])
    save_name = None
    
    #Training
    multiIm, _, _, _, fat_vector, meat_vector = load_day_data(train_day)
    t, best_band = train_threshold_value(fat_vector, meat_vector)
    Sigma_inv, mu_fat, mu_meat = train_multivariate_linear_discriminant(multiIm, fat_vector, meat_vector)
    
    err = np.zeros((np.size(compare_days),2))
    
    # Computation
    for j,i in enumerate(compare_days):
        if print_error: print("\nDay " + str(i))
        multiIm, true_fat, true_meat, index_background, _, _ = load_day_data(i)
        index_fat_tv, index_meat_tv = compute_threshold_value(multiIm,t, best_band)
        errorrate_tv = compute_errorrate(true_fat, true_meat, index_fat_tv, index_meat_tv, method_name = "TV"*int(print_error))
        index_fat_mld, index_meat_mld = compute_multivariate_linear_discriminant(multiIm, Sigma_inv, mu_fat, mu_meat)
        errorrate_mld = compute_errorrate(true_fat, true_meat, index_fat_mld, index_meat_mld, method_name = "MLD"*int(print_error))
        
        err[j,:] = [errorrate_tv,errorrate_mld]
        
        # Evaluation
        if print_error: print("The best method is " + "TV"*int(errorrate_tv < errorrate_mld) + "MLD"*int(errorrate_tv > errorrate_mld))
        
        if plot:
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
        result[j,:,:] = alpha(i,liste, plot = True, cmap = "seismic", save_fig=True, show_fig = False, print_error = False)
    
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