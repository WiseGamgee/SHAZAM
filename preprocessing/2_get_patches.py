import os
import yaml
from preprocessing_helpers import chop_images_in_directory, create_rgb


def main():
    # Import test configuration file from local directory
    with open("0_preprocessing_config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
    directory = config["DATA_DIR"]

    # Create training patch directory if it doesn't exist
    if os.path.exists(rf"{directory}\train_patched") is False:
        os.makedirs(rf"{directory}\train_patched")

    # Chop training images into smaller patches
    print("Chopping training images into smaller patches...")
    chop_images_in_directory(data_dir=rf"{directory}\train",
                             save_dir=rf"{directory}\train_patched",
                             img_size=config["IMG_SIZE"],
                             patch_size=config["PATCH_SIZE"])

    # Create corresponding RGB images for every patch
    print("Generating RGB patches...")
    create_rgb(data_dir=rf"{directory}\train_patched",
               save_dir=rf"{directory}\train_patched",
               rgb_bands=[2, 1, 0])


    # Create test patch directory if it doesn't exist
    if os.path.exists(rf"{directory}\test_patched") is False:
        os.makedirs(rf"{directory}\test_patched")

    # Chop test images into smaller patches
    print("Chopping test images into smaller patches...")
    chop_images_in_directory(data_dir=rf"{directory}\test",
                             save_dir=rf"{directory}\test_patched",
                             img_size=config["IMG_SIZE"],
                             patch_size=config["PATCH_SIZE"])

    # Create corresponding RGB images for every patch
    print("Generating RGB patches...")
    create_rgb(data_dir=rf"{directory}\test_patched",
               save_dir=rf"{directory}\test_patched",
               rgb_bands=[2, 1, 0])


if __name__ == '__main__':
    main()
