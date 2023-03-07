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

def compare_train(image1, image2 = 0, image3 = 0, day = 1, title="Titel", im_title = ["Original", "Threshold Value", "LDA without PN", "LDA with PA"], cmap = None, save_fig = None, show_fig = True):
    
    if cmap == None: cmap = "viridis"
    image = np.ones(4, dtype = int)
    if np.size(image2) < 4:
        #imagemld = np.zeros(np.shape(imagetv))
        image[2] = 0
    if np.size(image3) < 4: 
        #imagepi = np.zeros(np.shape(imagetv))
        image[3] = 0
    
    n = np.sum(image)

    try:
        color_im = skimage.io.imread("Salami/color_day" + "0"*(day<10) + str(day) + ".png")
    except FileNotFoundError:
        print("No image found for day " + str(day))
        color_im = np.ones(np.shape(image1))

    fig, ax = plt.subplots(1,n, figsize=(4*n,4.5), sharey=True)
    fig.subplots_adjust(wspace=0, hspace=10)
    fig.suptitle(title, fontsize = 16)
    ax[0].imshow(color_im, cmap = cmap)
    ax[0].set_title(im_title[0])
    ax[1].imshow(image1, cmap = cmap)
    ax[1].set_title(im_title[1])
    if image[2]:
        ax[2].imshow(image2, cmap = cmap)
        ax[2].set_title(im_title[2])
    if image[3]:
        ax[2 + image[2]].imshow(image3, cmap = cmap)
        ax[2 + image[2]].set_title(im_title[3])

    if save_fig:
        plt.draw()
        plt.savefig("Salami/plots/" + save_fig + ".png", dpi = 300)
    
    if show_fig: plt.pause(2)
    
    plt.close(fig)