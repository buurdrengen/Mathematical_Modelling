import numpy as np
from skimage.draw import disk
import matplotlib.pyplot as plt
from scipy.io import savemat
from scipy.ndimage import gaussian_filter
import time

def phantom(n_lead, n_steel, ring_count=50, wood_type = 'Beech', N=5000, r=20, inv_sigma_ring = 1.25, inv_sigma_log=4, plot = False, seed = 785462783):
    print("Generating Phantom... ", end='', flush=True)
    t =  time.time()
    shape = (N,N)
    img = np.zeros(shape)
    np.random.seed(seed)
    
    # Define tree rings
    rings = np.linspace(N,0,ring_count+1)
    # Add some variation to ring size
    d = N/(2*inv_sigma_ring*ring_count)
    variation = np.random.uniform(-d,d,ring_count)
    rings[1:] += variation; rings[-1] = 0
    alpha_wood = {'Beech': 2.5e-3, 'Fir': 1.7e-3}
    # Check wood type
    if not wood_type in alpha_wood.keys():
        raise Exception(f'Coefficient for Wood Type {wood_type} is not defined!')
            
    # Define other parameters
    alpha = [2*alpha_wood[wood_type]/3, 4*alpha_wood[wood_type]/3, 6.5e-1, 3.43] # Attenuation constants: [light wood, dark wood, iron ,lead]
    sigma = N/(4*inv_sigma_log*ring_count) # Standard deviation of Gaussian Blur
    
    # Create the log
    for i,R in enumerate(rings):
        if R == 0: break
        rr, cc = disk((N//2, N//2), R//2, shape=shape)
        img[rr,cc] = alpha[i%2]
        
    # Create the smooth transition
    blur_img = np.copy(img)
    blur_img[blur_img == 0] = alpha[0]
    blur_img = gaussian_filter(blur_img, sigma=sigma, mode='nearest')
    img[img != 0] = blur_img[img != 0]
    
    # Create steel pellets inside the log
    for i in range(n_steel):
        x,y = 0,0
        while (x-N/2)**2 + (y-N/2)**2 + r**2 > N**2/4:
            x,y = np.random.randint(0,N+1,2)
            
        rr, cc = disk((x,y), r//1, shape=shape)
        img[rr,cc] = alpha[2]
    
    # Create lead pellets inside the wood
    for i in range(n_lead):
        x,y = 0,0
        while (x-N/2)**2 + (y-N/2)**2 + r**2 > N**2/4:
            x,y = np.random.randint(0,N+1,2)
            
        rr, cc = disk((x,y), r//1, shape=shape)
        img[rr,cc] = alpha[3]
        
    print(f"Done in {time.time() - t:0.2f} seconds")
    if plot:
        class_img = np.copy(img)
        plt.imshow(class_img, vmin = 0, vmax = 3e-2)
        plt.show()
    return img
    
if __name__ == "__main__":
    im = phantom(2,2, ring_count = 50, plot = True, wood_type='Oak')
    # mdict = {'im': im}
    # savemat("new_testimage.mat",mdict, do_compression=True)
    
