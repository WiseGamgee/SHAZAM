import os
import yaml
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def delete_lowres_sen2_channels(directory):
    # Loop through all files in the directory
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith('.npy'):
            # Construct the full path to the file
            filepath = os.path.join(directory, filename)

            # Load the .npy file
            data = np.load(filepath)

            # Check the dimensions of the array
            if data.ndim == 3 and data.shape[2] == 13:
                # Remove the 1st, 10th and 13th channels
                data = np.delete(data, [0, 9, 12], axis=2)

                # Save the modified array back to the original file
                np.save(filepath, data)
            else:
                print(f"Skipping file {filename} due to incompatible dimensions.")
    print("Processing complete.")


def main():
    # Import test configuration file from local directory
    with open("0_preprocessing_config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
    directory = config["DATA_DIR"]

    print("Processing training images...")
    delete_lowres_sen2_channels(rf"{directory}\train")

    print("Processing test images...")
    delete_lowres_sen2_channels(rf"{directory}\test")

if __name__ == '__main__':
    main()


# # Construct the full path to the file
# filename = os.listdir(rf"{directory}\train")[0]
# filepath = os.path.join(rf"{directory}\train", filename)

# # Load the .npy file
# data = np.load(filepath)
# print(data.shape)
#
# # Number of channels
# channels = data.shape[2]
#
# # Create subplots (1 row, 'channels' columns)
# fig, axs = plt.subplots(1, channels, figsize=(15, 5))
#
# # Plot each channel
# for i in range(channels):
#     axs[i].imshow(data[:16, :16, i], cmap='gray')
#     axs[i].set_title(f'Channel {i+1}')
#     axs[i].axis('off')  # Hide axis
#
# # Display the plot
# plt.tight_layout()
# plt.show()
