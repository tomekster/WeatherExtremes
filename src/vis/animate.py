import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
import numpy as np
import imageio

def temperature_array_to_video(temperature_data, output_video='temperature_video.mp4', fps=10):
    """
    Convert a 3D NumPy array of daily temperature data into a video.
    
    Args:
    - temperature_data (numpy array): Array of shape (365, 721, 1440), where each (721, 1440)
                                      slice represents daily average temperatures on Earth.
    - output_video (str): Name of the output video file.
    - fps (int): Frames per second for the video.
    
    Returns:
    - None
    """
    # Create a temporary directory to save images
    if not os.path.exists('temp_images'):
        os.makedirs('temp_images')
    
    height, width = temperature_data.shape[1], temperature_data.shape[2]
    
    # Loop through each day's temperature data and generate a plot
    print("Processing days...")
    for day in tqdm(list(range(temperature_data.shape[0]))):
        plt.figure(figsize=(10, 5))
        
        # Plot the temperature data for that day using a heatmap
        plt.imshow(temperature_data[day], cmap='coolwarm', extent=[-180, 180, -90, 90])
        plt.title(f"Day {day + 1} of 365")
        plt.colorbar(label='Temperature (°C)')
        
        # Save the figure as an image
        plt.savefig(f'temp_images/day_{day:03d}.png')
        plt.close()
    
    # Create a video from the saved images
    img_array = []
    
    for day in range(temperature_data.shape[0]):
        img = cv2.imread(f'temp_images/day_{day:03d}.png')
        img_array.append(img)
    
    # Define video codec and output file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 format
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    for img in img_array:
        video.write(img)
    
    video.release()
    
    # Clean up temporary images
    # for day in range(temperature_data.shape[0]):
    #     os.remove(f'temp_images/day_{day:03d}.png')
    
    # os.rmdir('temp_images')
    
    print(f"Video saved as {output_video}")

def temperature_array_to_video2(temperature_data, output_video='temperature_video.mp4', fps=10):
    """
    Convert a 3D NumPy array of daily temperature data into a video.
    
    Args:
    - temperature_data (numpy array): Array of shape (365, 721, 1440), where each (721, 1440)
                                      slice represents daily average temperatures on Earth.
    - output_video (str): Name of the output video file.
    - fps (int): Frames per second for the video.
    
    Returns:
    - None
    """
    # Create a temporary directory to save images
    if not os.path.exists('temp_images'):
        os.makedirs('temp_images')
    
    images = []
    
    # Loop through each day's temperature data and generate a plot
    for day in range(temperature_data.shape[0]):
        plt.figure(figsize=(10, 5))
        
        # Plot the temperature data for that day using a heatmap
        plt.imshow(temperature_data[day], cmap='coolwarm', extent=[-180, 180, -90, 90])
        plt.title(f"Day {day + 1} of 365")
        plt.colorbar(label='Temperature (°C)')
        
        # Save the figure as an image
        image_path = f'temp_images/day_{day:03d}.png'
        plt.savefig(image_path)
        plt.close()
        
        # Append the saved image to the list for video generation
        images.append(imageio.imread(image_path))
    
    # Create the video using imageio
    imageio.mimsave(output_video, images, fps=fps)

    # Clean up temporary images
    # for image_file in os.listdir('temp_images'):
    #     os.remove(os.path.join('temp_images', image_file))
    # os.rmdir('temp_images')
    
    print(f"Video saved as {output_video}")

    
temperature_data = np.load('t2m_1960_1990_99_perc_agg5_boost_5.npy')
temperature_array_to_video2(temperature_data)
