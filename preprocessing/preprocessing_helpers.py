import os
import tifffile
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def chop_single_image(img_dir, save_dir, img_size, patch_size):
    # Ensure that patch size is a factor of img size
    assert img_size % patch_size == 0

    # Calculate the number of patches along each axis
    axis_ratio = img_size // patch_size

    # Load image to be chopped
    img = np.load(img_dir, allow_pickle=True)

    # Create patches
    patch_count = 0

    # Loop through width
    for i in range(axis_ratio):
        x_0, x_1 = i * patch_size, (i + 1) * patch_size

        # Loop through height
        for j in range(axis_ratio):
            y_0, y_1 = j * patch_size, (j + 1) * patch_size

            # Extract current patch and save
            patch = img[x_0: x_1, y_0: y_1]
            save_dir = os.path.splitext(save_dir)[0]
            np.save(file=f"{save_dir}_patch_{i}_{j}",
                    arr=patch)
            patch_count += 1


def chop_images_in_directory(data_dir,
                             save_dir,
                             img_size,
                             patch_size):
    # Get list of all samples in data directory
    img_filenames = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

    # Loop through all images in directory and chop them into patches
    for img_name in tqdm(img_filenames):
        in_dir = os.path.join(data_dir, img_name)
        out_dir = os.path.join(save_dir, img_name)

        # Chop image into patches
        chop_single_image(img_dir=in_dir,
                          save_dir=out_dir,
                          img_size=img_size,
                          patch_size=patch_size)


def create_rgb(data_dir,
               save_dir,
               rgb_bands: list = None):
    # Get list of all image filenames in directory
    if rgb_bands is None:
        rgb_bands = [3, 2, 1]

    # Get list of all samples in data directory
    img_filenames = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

    # Load each image and save RGB component
    for img_name in tqdm(img_filenames):
        img = np.load(f"{data_dir}/{img_name}", allow_pickle=True)
        rgb_img = img[..., rgb_bands]

        rgb_img = (rgb_img - np.min(rgb_img)) / (np.max(rgb_img) - np.min(rgb_img))
        img_dir = f"{save_dir}/{os.path.splitext(img_name)[0]}.png"
        plt.imsave(img_dir,
                   arr=rgb_img,
                   format="png")


def create_index(data_dir,
                 save_dir,
                 band_1,
                 band_2,
                 cmap="jet"):
    # Get list of all samples in data directory
    img_filenames = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

    # Load each image and save RGB component
    for img_name in tqdm(img_filenames):
        img = np.load(f"{data_dir}/{img_name}", allow_pickle=True)
        index_img = (img[..., band_2] - img[..., band_1]) / (img[..., band_2] + img[..., band_1] +1e-10)
        img_dir = f"{save_dir}/{os.path.splitext(img_name)[0]}_index.png"
        plt.imsave(img_dir,
                   arr=index_img,
                   format="png",
                   cmap=cmap)


def tiff2numpy(data_dir,
               save_dir):
    # Get list of all samples in data directory
    img_filenames = [f for f in os.listdir(data_dir) if f.endswith('.tif')]

    # Load each image and save RGB component
    for img_name in tqdm(img_filenames):
        img = tifffile.imread(f"{data_dir}/{img_name}").astype(np.uint8)
        np.save(file=f"{save_dir}/{os.path.splitext(img_name)[0]}",
                arr=img)
        scaled_img = (img - np.min(img)) / (np.max(img) - np.min(img))
        plt.imsave(f"{save_dir}/{img_name}",
                   arr=scaled_img[..., [3,2,1]],
                   format="png")
