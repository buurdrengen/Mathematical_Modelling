import numpy as np
import matplotlib.pyplot as plt
import skimage

spectre = np.array([410, 438, 450, 468, 502, 519, 572, 591, 625, 639, 653, 695, 835, 863, 880, 913, 929, 940, 955])

def compare_image(image, day = 1, title="Titel"):

    # if np.shape(image1) != np.shape(image2):
    #     print("Shapes of images mismatch!")
    #     return None

    str_day = "0"*(day<10) + str(day)
    
    try:
        color_im = skimage.io.imread("Salami/color_day" + str_day + ".png")
    except FileNotFoundError:
        print("No image found for day " + str(day))
        color_im = np.ones(np.shape(image))

    fig, [ax1,ax2] = plt.subplots(1,2, figsize=(8,4))
    fig.suptitle(title)
    ax1.imshow(color_im)
    ax2.imshow(image)
    plt.show()



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
