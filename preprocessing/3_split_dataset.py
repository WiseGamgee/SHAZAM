import os
import tqdm
import yaml
import shutil
import random


def move_files(source_dir: str, target_dir: str, fraction_moved: float):
    # Ensure target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Get list of all .npy files in the source directory
    npy_files = [f for f in os.listdir(source_dir) if f.endswith('.npy')]

    # Ensure we have PNG matches for each NPY file
    matched_files = []
    for npy_file in npy_files:
        png_file = npy_file.replace('.npy', '.png')
        if os.path.exists(os.path.join(source_dir, png_file)):
            matched_files.append((npy_file, png_file))

    # Calculate the number of files to move
    num_files_to_move = int(len(matched_files) * fraction_moved)

    # Randomly select files to move
    files_to_move = random.sample(matched_files, num_files_to_move)

    # Move the selected files
    for npy_file, png_file in tqdm.tqdm(files_to_move):
        shutil.move(os.path.join(source_dir, npy_file), os.path.join(target_dir, npy_file))
        shutil.move(os.path.join(source_dir, png_file), os.path.join(target_dir, png_file))

    print(f"Moved {num_files_to_move} pairs of files from {source_dir} to {target_dir}.")


def main():
    # Import test configuration file from local directory
    with open("0_preprocessing_config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
    directory = config["DATA_DIR"]

    print("Creating validation set...")
    move_files(source_dir=rf"{directory}\train_patched",
               target_dir=rf"{directory}\val_patched",
               fraction_moved=0.1)


if __name__ == '__main__':
    main()