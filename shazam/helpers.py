import os
import re
import tqdm
import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt


def load_img_from_file(img_folder, img_name, get_date=True):
    # Load image from file
    img_path = f"{img_folder}/{img_name}"
    img = np.load(img_path)

    # Swap hwc to chw and convert to torch tensor
    img = np.moveaxis(img, 2, 0)
    img = torch.from_numpy(img).float()

    # Get temporal information
    if get_date:
        date = re.search(r'\d{4}-\d{2}-\d{2}', img_name)[0]
    else:
        date = None
    return img, date


def date_to_polar_period(date_str, time_period):
    # Parse the date string into a datetime object
    date = datetime.datetime.strptime(date_str, "%Y-%m-%d")

    if time_period == 'day':
        # Return the day of the year (0-indexed)
        t_int = date.timetuple().tm_yday - 1
        total_periods = 365 + (
            1 if date.year % 4 == 0 and (date.year % 100 != 0 or date.year % 400 == 0) else 0)  # handle leap years

    elif time_period == 'week':
        # Return the ISO week number of the year (0-indexed)
        t_int = date.isocalendar()[1] - 1
        total_periods = 52 + (
            1 if datetime.date(date.year, 12, 28).isocalendar()[1] == 53 else 0)  # handle weeks in a year

    elif time_period == 'fortnight':
        # Calculate the fortnight number (two-week period, 0-indexed)
        week_number = date.isocalendar()[1]
        t_int = (week_number + 1) // 2 - 1
        total_periods = 26 + (1 if (datetime.date(date.year, 12, 28).isocalendar()[
                                        1] == 53 and week_number % 2 == 1) else 0)  # handle fortnights in a year

    elif time_period == 'month':
        # Return the month number (0-indexed)
        t_int = date.month - 1
        total_periods = 12

    elif time_period == 'season':
        # Calculate the season of the year (0-indexed)
        if date.month in [12, 1, 2]:
            t_int = 0  # Summer
        elif date.month in [3, 4, 5]:
            t_int = 1  # Autumn
        elif date.month in [6, 7, 8]:
            t_int = 2  # Winter
        elif date.month in [9, 10, 11]:
            t_int = 3  # Spring
        total_periods = 4

    else:
        raise ValueError("Invalid period. Choose from 'day', 'week', 'fortnight', 'month', 'season'.")

    # Convert the period integer to polar coordinates (sin, cos)
    theta = (2 * np.pi * t_int) / total_periods
    t_sin = np.sin(theta)
    t_cos = np.cos(theta)
    t = torch.Tensor([t_sin, t_cos]).float()

    # Return time variable
    return t


def patch_name_to_coords(patch_name, patches_per_dim):
    # Get row and column of patch in image
    c1, c2 = map(int, patch_name.split("patch_")[1].split(".npy")[0].split('_'))

    # Scale between 0 and 1 and return as tensor
    c = (torch.Tensor([c1, c2]) / (patches_per_dim - 1)).float()
    return c


def calculate_mean_image(data_dir, img_channels, num_patches=None, patch_size=None):
    """
    Calculate the mean image representation of all .npy images in the given directory.

    Parameters:
    - directory (str): Path to the directory containing .npy images.

    Returns:
    - mean_image (numpy.ndarray): The mean image representation.
    """
    # List all .npy files in the directory
    img_filenames = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

    if not img_filenames:
        raise ValueError("No .npy files found in the directory")

    # Initialize the sum of images
    img_sum = np.zeros(shape=(num_patches, num_patches, patch_size, patch_size, img_channels))
    img_count = np.zeros(shape=(num_patches, num_patches, patch_size, patch_size, img_channels))

    # Loop through each .npy file
    for img_name in tqdm.tqdm(img_filenames):
        # Load the image
        img = np.load(os.path.join(data_dir, img_name))

        # Add the image to the sum
        row, col = map(int, img_name.split("patch_")[1].split(".npy")[0].split('_'))
        img_sum[row, col] += img
        img_count[row, col] += 1

    # Calculate the mean for all patches
    patch_means = img_sum / img_count

    # Initialize the full image with zeros
    mean_img = np.zeros(shape=(num_patches * patch_size, num_patches * patch_size, img_channels),
                        dtype=patch_means.dtype)

    # Fill the full image with patches
    for row in range(num_patches):
        for col in range(num_patches):
            mean_img[row * patch_size:(row + 1) * patch_size, col * patch_size:(col + 1) * patch_size, :] = patch_means[
                row, col]
    return patch_means


def normalise_image_tensor(img, channels_min, channels_max, clamp=True):
    """
    Normalize a torch tensor image given per-channel min and max values.

    Parameters:
    - image (torch.Tensor): Input image tensor with shape (c, h, w).
    - min_vals (torch.Tensor): 1D tensor of minimum values for each channel with shape (c,).
    - max_vals (torch.Tensor): 1D tensor of maximum values for each channel with shape (c,).

    Returns:
    - torch.Tensor: Normalized image tensor with shape (c, h, w).
    """
    # Ensure min_vals and max_vals are 1D tensors
    channels_min = torch.Tensor(channels_min).view(-1, 1, 1)
    channels_max = torch.Tensor(channels_max).view(-1, 1, 1)

    # Normalize the image tensor
    img_normalised = (img - channels_min) / (channels_max - channels_min)
    if clamp:
        return torch.clamp(img_normalised, 0, 1)
    else:
        return img_normalised


def save_tensor_as_image(filename, tensor, rgb_bands, v_min=None, v_max=None, normalise=True):
    # Get number of dimensions in tensor
    dims = tensor.dim()
    rgb = None
    if dims == 3:
        # Assumed tensor is in chw format
        rgb = tensor[rgb_bands].permute(1, 2, 0).cpu().detach().numpy()
    elif dims == 2:
        # Assumed tensor is in hw format
        rgb = tensor.cpu().detach().numpy()
    else:
        raise ValueError(f"{dims} dimensions found. Please check tensor structure.")

    # Normalise and save image
    if normalise:
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    plt.imsave(filename, arr=rgb, format="png", vmin=v_min, vmax=v_max, cmap="hot")
