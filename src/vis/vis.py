import numpy as np
import matplotlib.pyplot as plt

def plot_and_save_arrays(array1, array2, array3, array4, filename='plot.png'):
    # Find the global minimum and maximum values across all arrays to set the color scale
    vmin = min(np.min(array1), np.min(array2), np.min(array3), np.min(array4))
    vmax = max(np.max(array1), np.max(array2), np.max(array3), np.max(array4))
    
    # Create a figure with 4 subplots (2x2 grid)
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Plot each array in a separate subplot with a shared color scale
    im1 = axs[0, 0].imshow(array1, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[0, 0].set_title('Array 1')

    im2 = axs[0, 1].imshow(array2, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[0, 1].set_title('Array 2')

    im3 = axs[1, 0].imshow(array3, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[1, 0].set_title('Array 3')

    im4 = axs[1, 1].imshow(array4, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[1, 1].set_title('Array 4')

    # Add a colorbar that is shared across all subplots
    fig.colorbar(im1, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
    
    # Adjust layout
    plt.tight_layout()

    # Save the plot to the specified file
    plt.savefig(filename)

    # Optionally show the plot
    plt.show()
