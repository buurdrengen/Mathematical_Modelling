
def final_func(angle_no, 
                p, 
                res, 
                air_tree_separator, tree_steel_separator, steel_lead_separator,
                confidence = 2, 
                sample_size=20, 
                noise_limit = [1e-4, 1e-3], 
                noise_size = 40, 
                class_errors=True):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.io import loadmat
    # from scipy.linalg import solve
    # from scipy.stats import alpha
    from skimage.measure import block_reduce
    import paralleltomo # [A,theta,p,d] = paralleltomo(N,theta,p,d)
    
    # Load the image with lead and steel shot
    data = np.array(loadmat("Exam_02526/new_testImage.mat")['im']) #Pixel size: 0.1 mm, 5000x5000 pixels

    base_array = np.zeros(np.shape(data))   
    known_wood = np.copy(base_array); known_wood[data == np.unique(data)[[1]]] = 1
    known_iron= np.copy(base_array); known_iron[data == np.unique(data)[[2]]] = 1
    known_lead = np.copy(base_array); known_lead[data == np.unique(data)[[3]]] = 1

    # Downsample image
    downsizing = int(5e3/res) # define downsizing constant
    downsized_im = block_reduce(data, block_size=(downsizing, downsizing), func=np.mean) # downsample the image
    x = np.copy(downsized_im.flatten()) # flatten the attenuation constants

    downsized_known_wood = block_reduce(known_wood, block_size=downsizing, func=np.min).astype(int) # downsample the image
    downsized_known_iron = block_reduce(known_iron, block_size=downsizing, func=np.max).astype(int) # downsample the image
    downsized_known_lead = block_reduce(known_lead, block_size=downsizing, func=np.max).astype(int) # downsample the image

    # Find the separators based on the anotted masks
    x_imShape = x.reshape(np.shape(downsized_im)) 
    # Define the attenuation coefficient means and find the separators betweeen classes

    # number of each cell
    N_lead = np.sum(downsized_known_lead)
    N_iron = np.sum(downsized_known_iron)
    N_wood = np.sum(downsized_known_wood)

    
    print(f"the resolution of the image is {np.shape(downsized_im)}")

    # Initialize system matrix and stuff
    theta = np.array([np.linspace(0, 180, angle_no)]) 
    d = 0.5 # [m]
    N = downsized_im.shape[0]
    [A,theta,p,d] = paralleltomo(N, theta, p)


    # Load simulated forward projection 
    b = A @ x # Finding bj
    error_list = np.linspace(noise_limit[0], noise_limit[1], noise_size)
    wood_errors = np.zeros([np.size(error_list),sample_size])
    failed_to_detect_wood = -1
    failed_to_detect_metal = -1
    for i, noise in enumerate(error_list):
        for j in range(sample_size):
            np.random.seed(j)
            b_perturbed = b + np.random.normal(0, noise, size=np.shape(b)) # adding noise

            # Find the perturbed attenuation coefficients
            x_new = solve(A.T @ A, A.T @ b_perturbed, assume_a = "her")
            x_new = x_new.reshape(np.shape(downsized_im)) 

            # Find the index for the different classes
            air_index =                                 (x_new < air_tree_separator)
            tree_index = (x_new > air_tree_separator) & (x_new < tree_steel_separator)
            iron_index = (x_new > tree_steel_separator) & (x_new < steel_lead_separator)
            lead_index = (x_new > steel_lead_separator)

            # Find the false positive errors
            lead_error = np.sum(lead_index[downsized_known_lead == 1] == 0)/N_lead
            iron_error = np.sum(iron_index[downsized_known_iron == 1] == 0)/N_iron
            wood_error = np.sum(tree_index[downsized_known_wood == 1] == 0)/N_wood
            wood_errors[i,j] = (wood_error*N_wood + iron_error*N_iron + lead_error*N_lead)/(N_lead + N_iron + N_wood)

            # Find the values in the confusion matrix
            if noise == noise_limit[1]:
                lead_as_lead = np.sum(lead_index[downsized_known_lead == 1] == 1)
                iron_as_lead = np.sum(lead_index[downsized_known_iron == 1] == 1)
                wood_as_lead = np.sum(lead_index[downsized_known_wood == 1] == 1)

                iron_as_iron = np.sum(iron_index[downsized_known_iron == 1] == 1)
                lead_as_iron = np.sum(iron_index[downsized_known_lead == 1] == 1)
                wood_as_iron = np.sum(iron_index[downsized_known_wood == 1] == 1)

                wood_as_wood = np.sum(tree_index[downsized_known_wood == 1] == 1)
                iron_as_wood = np.sum(tree_index[downsized_known_iron == 1] == 1)
                lead_as_wood = np.sum(tree_index[downsized_known_lead == 1] == 1)

            
            if (wood_error > 0) & (failed_to_detect_wood == -1): failed_to_detect_wood = noise; print(f"FAIL: (at {i}) Failed To Detect Wood...")
            if (lead_error + iron_error > 0) & (failed_to_detect_metal == -1): failed_to_detect_metal = noise; print(f"FAIL: (at {i}) Failed To Detect Metal...")

        print(f"Error[{i}] is {np.mean(wood_errors[i,:])} pm {confidence*np.std(wood_errors[i,:])}")


    mean_error = np.mean(wood_errors,axis=1)
    std_error = np.std(wood_errors,axis=1)

    plt.fill_between(error_list, y1 = (mean_error + confidence*std_error)*100, y2 = (mean_error - confidence*std_error)*100, color="gray", lw=0, alpha=0.3, label="Confidence Interval")
    plt.plot(error_list, mean_error*100, label="Error Rate", lw=2)
    if failed_to_detect_wood >= 0: plt.axvline(failed_to_detect_wood, ls="--", color="orange", label="First False Positive")
    if failed_to_detect_metal >= 0: plt.axvline(failed_to_detect_metal, ls="--", color="red", label="First False Negative")
    plt.ylabel("Error Rate [%]", fontsize=14)
    plt.xlabel("Added noise level", fontsize=14)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(-4,-4), useMathText=True)
    plt.xlim([0,1e-3])
    plt.ylim([0,50])
    plt.tick_params(labelsize=11)
    plt.legend(fontsize=12)
    plt.grid()
    plt.title(f'Resolution: {500/res}X{500/res} [mm]\nSetup: {p} Rays, {angle_no} Angles and {sample_size} Samples', fontsize=16)
    plt.savefig(f"Exam_02526/img/res{res}.png", dpi=300)
    

    if class_errors==True:
            
        # Define a class tree to store the different classes identified. These are created from the last noise level.
        class_tree = np.ones(np.shape(x_new))
        class_tree[tree_index] = 2
        class_tree[iron_index] = 3
        class_tree[lead_index] = 4
        class_tree_true = np.ones(np.shape(x_new))
        class_tree_true[downsized_known_wood==1] = 2
        class_tree_true[downsized_known_iron==1] = 3
        class_tree_true[downsized_known_lead==1] = 4
        

        # Define a confusion matrix to be used for plotting
        confusion_matrix = np.array([
            [wood_as_wood, wood_as_iron, wood_as_lead]/N_wood,
            [iron_as_wood, iron_as_iron, iron_as_lead]/N_iron,
            [lead_as_wood, lead_as_iron, lead_as_lead]/N_lead
        ])

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,4))
        cmap = plt.get_cmap('viridis', 4) # Define the colormap
        im = ax1.imshow(class_tree, cmap=cmap, vmin=0.5, vmax=4.5)
        im = ax2.imshow(class_tree_true, cmap=cmap, vmin=0.5, vmax=4.5)
        fig.suptitle(f'Classification Comparison of Noise: {noise_limit[1]}')
        ax1.set_title('Modelled Classes')
        ax2.set_title('Real Classes')

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax, ticks=[1, 2, 3, 4]) # Define colorbar
        cbar.ax.set_yticklabels(['Air', 'Tree', 'Iron', 'Lead']) # Define tick labels
        plt.savefig(f"Exam_02526/img/classification_{res}.png")


        fig, ax3 = plt.subplots()
        sns.heatmap(
            confusion_matrix*100, 
            annot=True, 
            fmt='.2f', 
            ax=ax3, 
            linewidths=.05, 
            cbar_kws={'label':'Error Rate [%]'},
            xticklabels=['Wood', 'Iron', 'Lead'],
            yticklabels=['Wood', 'Iron', 'Lead']
        )

        plt.savefig(f"Exam_02526/img/confusion{res}.png", dpi=300)
        ax3.set_ylabel('Real Class')
        ax3.set_xlabel('Modelled Class')
        ax3.set_title('Confusion Matrix')
        plt.savefig(f"Exam_02526/img/confusion_{res}.png")

