import numpy as np
import matplotlib.pyplot as plt
import skimage

def compare_image(image, day=1, title="Titel"):

    # if np.shape(image1) != np.shape(image2):
    #     print("Shapes of images mismatch!")
    #     return None
    try:
        color_im = skimage.io.imread("Salami/color_day" + str(day) + ".png")
    except FileNotFoundError:
        print("No image found for selected day: " + str(day))
        color_im = np.ones(np.shape(image))

    fig, [ax1,ax2] = plt.subplots(1,2, figsize=(8,4))
    fig.suptitle(title)
    ax1.imshow(color_im)
    ax2.imshow(image)
    plt.show()



def compare_spectrum(mean_fat, sd_fat, mean_meat, sd_meat, signific = 2, ecolor = "black", capsize = 5, title = "Titel"):
    spectrum = np.array([410, 438, 450, 468, 502, 519, 572, 591, 625, 639, 653, 695, 835, 863, 880, 913, 929, 940, 955])

    plt.figure()
    plt.errorbar(spectrum, mean_fat, signific*sd_fat, capsize=capsize, ecolor=ecolor, label='fat', color='orange')
    plt.errorbar(spectrum ,mean_meat, signific*sd_meat, capsize=capsize, ecolor=ecolor,  label='meat', color='blue')
    plt.legend()
    plt.grid()
    plt.xlabel("Frequency [nm]")
    plt.ylabel("Mean Intensity")
    plt.title(title)
    plt.show()
