import numpy as np
import matplotlib.pyplot as plt
import skimage

spectre = np.array([410, 438, 450, 468, 502, 519, 572, 591, 625, 639, 653, 695, 835, 863, 880, 913, 929, 940, 955])

# ---------------------------------------------------------------------

def compare_image(image, day = 1, title="Titel"):

    try:
        color_im = skimage.io.imread("Salami/color_day" + "0"*(day<10) + str(day) + ".png")
    except FileNotFoundError:
        print("No image found for day " + str(day))
        color_im = np.ones(np.shape(image))

    fig, [ax1,ax2] = plt.subplots(1,2, figsize=(8,4))
    fig.suptitle(title)
    ax1.imshow(color_im)
    ax2.imshow(image)
    plt.show()

# ---------------------------------------------------------------------

def compare_spectrum(mean_fat, sd_fat, mean_meat, sd_meat, signific = 2, ecolor = ["black", "black"], capsize = 5, color = ["orange", "blue"], spectrum = spectre, title = "Titel", xtick_align = False):
    

    plt.figure()
    plt.errorbar(spectrum, mean_fat, signific*sd_fat, capsize=capsize, ecolor=ecolor[0], label='fat', color=color[0])
    plt.errorbar(spectrum ,mean_meat, signific*sd_meat, capsize=capsize, ecolor=ecolor[1],  label='meat', color=color[1])
    plt.legend()
    plt.grid()
    plt.xlabel("Frequency [nm]")
    plt.ylabel("Mean Intensity")
    plt.title(title)
    if xtick_align: plt.xticks(spectrum)
    plt.show()

# ---------------------------------------------------------------------

def compare_train(imagetv, imagemld, day = 1, title="Titel", im_title = ["Original", "Threshold Value", "Multivariate Linear Discriminant"], cmap = None, save_fig = False):
    
    if cmap == None: cmap = "viridis"
    
    try:
        color_im = skimage.io.imread("Salami/color_day" + "0"*(day<10) + str(day) + ".png")
    except FileNotFoundError:
        print("No image found for day " + str(day))
        color_im = np.ones(np.shape(imagetv))

    fig, [ax1,ax2,ax3] = plt.subplots(1,3, figsize=(12,4))
    fig.suptitle(title)
    ax1.imshow(color_im, cmap = cmap)
    ax1.set_title(im_title[0])
    ax2.imshow(imagetv, cmap = cmap)
    ax2.set_title(im_title[1])
    ax3.imshow(imagemld, cmap = cmap)
    ax3.set_title(im_title[2])
    plt.show()
    
    if save_fig:
        None
    
    plt.close(fig)