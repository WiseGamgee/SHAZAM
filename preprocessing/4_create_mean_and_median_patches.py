import os
import tqdm
import yaml
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def get_mean_patches(input_directory, output_directory, rgb_bands):
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    patch_sums = defaultdict(lambda: np.zeros((0, 0, 0), dtype=np.float64))
    patch_counts = defaultdict(int)

    # Loop through all files in the input directory
    for filename in tqdm.tqdm(os.listdir(input_directory)):
        if filename.endswith('.npy'):
            # Extract row and column numbers from filename
            parts = filename.split('_')
            row = int(parts[-2])
            col = int(parts[-1].split('.')[0])

            # Construct the full path to the file
            filepath = os.path.join(input_directory, filename)

            # Load the .npy file
            x = np.load(filepath)

            # Initialize the shape of the patches if this is the first file
            if patch_sums[(row, col)].shape == (0, 0, 0):
                patch_sums[(row, col)] = np.zeros_like(x, dtype=np.float64)

            # Accumulate the sum of the patches
            patch_sums[(row, col)] += x
            patch_counts[(row, col)] += 1

    # Calculate the mean patch for each (row, col) and save it
    for (row, col) in patch_sums:
        mean_patch = patch_sums[(row, col)] / patch_counts[(row, col)]
        output_filename = f"mean_patch_{row}_{col}"
        output_filepath = os.path.join(output_directory, output_filename)
        np.save(output_filepath, mean_patch)

        # Save RGB image for mean patch
        rgb = mean_patch[..., rgb_bands]
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
        img_dir = f"{output_filepath}.png"
        plt.imsave(img_dir,
                   arr=rgb,
                   format="png")


def get_median_patches(input_directory, output_directory, rgb_bands):
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Dictionary to store patches based on their (row, col) keys
    patches = defaultdict(list)

    # Loop through all files in the input directory
    for filename in tqdm.tqdm(os.listdir(input_directory)):
        if filename.endswith('.npy'):
            # Extract row and column numbers from filename
            parts = filename.split('_')
            row = int(parts[-2])
            col = int(parts[-1].split('.')[0])

            # Construct the full path to the file
            filepath = os.path.join(input_directory, filename)

            # Load the .npy file
            data = np.load(filepath)

            # Store the patch data
            patches[(row, col)].append(data)

    # Calculate the median patch for each (row, col) and save it
    for (row, col), patch_list in patches.items():
        # Stack the patches along a new dimension
        stacked_patches = np.stack(patch_list, axis=0)

        # Calculate the median along the new dimension
        median_patch = np.median(stacked_patches, axis=0)

        # Construct the output filename
        output_filename = f"median_patch_{row}_{col}"
        output_filepath = os.path.join(output_directory, output_filename)

        # Save the median patch
        np.save(output_filepath, median_patch)

        # Save RGB image for median patch
        rgb = median_patch[..., rgb_bands]
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
        img_dir = f"{output_filepath}.png"
        plt.imsave(img_dir,
                   arr=rgb,
                   format="png")


def reconstruct_full_image(mean_patches):
    # Determine the maximum row and column indices
    max_row = max(row for row, col in mean_patches.keys())
    max_col = max(col for row, col in mean_patches.keys())

    # Determine the patch shape
    sample_patch = next(iter(mean_patches.values()))
    patch_height, patch_width, patch_channels = sample_patch.shape

    # Initialize the full image array
    full_image_height = (max_row + 1) * patch_height
    full_image_width = (max_col + 1) * patch_width
    full_image = np.zeros((full_image_height, full_image_width, patch_channels), dtype=np.float64)

    # Loop through the mean patches and place them in the full image
    for (row, col), patch in mean_patches.items():
        start_row = row * patch_height
        start_col = col * patch_width
        full_image[start_row:start_row + patch_height, start_col:start_col + patch_width, :] = patch

    return full_image


def main():
    # Import test configuration file from local directory
    with open("0_preprocessing_config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
    directory = config["DATA_DIR"]

    patch_dir = rf"{directory}\train_patched"
    mean_dir = rf"{directory}\mean_patched"
    median_dir = rf"{directory}\median_patched"
    get_mean_patches(patch_dir, mean_dir, rgb_bands=[2,1,0])
    get_median_patches(patch_dir, median_dir, rgb_bands=[2,1,0])


if __name__ == '__main__':
    main()
