import os
import re
import csv
import tqdm
import yaml
import random
import datetime
import numpy as np


def get_training_stats(n_channels, data_dir, test_start_date=None, data_frac=1.0):
    source_dir = rf"{data_dir}\train_patched"
    # Get img filenames
    img_filenames = []
    for _, _, img_files in os.walk(source_dir):
        for img_name in sorted(img_files):
            if img_name.endswith(".npy"):
                img_filenames.append(img_name)

    if test_start_date:
        # Get datetime of starting test time
        test_start_dt = datetime.datetime.strptime(test_start_date, '%Y-%m-%d')

        # Strip datetime from image filenames
        train_filenames = []
        for img_filename in img_filenames:
            date = re.search(r'\d{4}-\d{2}-\d{2}', img_filename)[0]
            current_dt = datetime.datetime.strptime(date, '%Y-%m-%d')

            # Check if image is in training window
            if current_dt < test_start_dt:
                train_filenames.append(img_filename)
            else:
                continue
    else:
        train_filenames = img_filenames

    # Loop through all images and append pixels to list
    img = None
    pixels_list = []
    random.shuffle(train_filenames)
    num_samples = int(len(train_filenames) * data_frac)
    for i in tqdm.tqdm(range(num_samples)):
        # Load current image array
        img_path = os.path.join(source_dir, train_filenames[i])
        img = np.load(img_path)

        # Flatten all pixels into vector
        pixel_vector = img.reshape((-1, img.shape[-1]))
        pixels_list.append(pixel_vector)

    # Convert list to array of pixels
    pixel_arr = np.array(pixels_list).reshape((-1, img.shape[-1]))

    # Get stats across entire dataset
    train_mean = pixel_arr.mean()
    train_std = pixel_arr.std()
    train_med = np.median(pixel_arr)
    train_min = pixel_arr.min()
    train_max = pixel_arr.max()
    train_perc1 = np.percentile(pixel_arr, q=1)
    train_perc2 = np.percentile(pixel_arr, q=2)
    train_perc98 = np.percentile(pixel_arr, q=98)
    train_perc99 = np.percentile(pixel_arr, q=99)

    # Create dictionary storing aggregate_stats stats
    aggregate_stats = {"train_mean": np.round(train_mean, 5),
                       "train_std": np.round(train_std, 5),
                       "train_med": np.round(train_med, 5),
                       "train_min": np.round(train_min, 5),
                       "train_max": np.round(train_max, 5),
                       "train_1%": np.round(train_perc1, 5),
                       "train_2%": np.round(train_perc2, 5),
                       "train_98%": np.round(train_perc98, 5),
                       "train_99%": np.round(train_perc99, 5)}

    # Write dictionary to csv file
    with open(rf'{data_dir}\training_stats.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in aggregate_stats.items():
            writer.writerow([key, value])

    # # Get stats across all channels
    # train_mean = pixel_arr.mean(axis=0)
    # train_std = pixel_arr.std(axis=0)
    # train_med = np.median(pixel_arr, axis=0)
    # train_min = pixel_arr.min(axis=0)
    # train_max = pixel_arr.max(axis=0)
    # train_perc1 = np.percentile(pixel_arr, q=1, axis=0)
    # train_perc2 = np.percentile(pixel_arr, q=2, axis=0)
    # train_perc98 = np.percentile(pixel_arr, q=98, axis=0)
    # train_perc99 = np.percentile(pixel_arr, q=99, axis=0)
    #
    # # Create dictionary storing band stats
    # band_stats = {"channel_indices": np.arange(n_channels),
    #               "train_mean": np.round(train_mean, 5),
    #               "train_std": np.round(train_std, 5),
    #               "train_med": np.round(train_med, 5),
    #               "train_min": np.round(train_min, 5),
    #               "train_max": np.round(train_max, 5),
    #               "train_1%": np.round(train_perc1, 5),
    #               "train_2%": np.round(train_perc2, 5),
    #               "train_98%": np.round(train_perc98, 5),
    #               "train_99%": np.round(train_perc99, 5)}
    #
    # # Write dictionary to csv file
    # with open(f'band_stats.csv', 'w') as csv_file:
    #     writer = csv.writer(csv_file)
    #     for key, value in band_stats.items():
    #         writer.writerow([key, value])
    return None


def main():
    # Import test configuration file from local directory
    with open("0_preprocessing_config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
    directory = config["DATA_DIR"]

    # Get summary statistics (used for normalisation or other means)
    get_training_stats(n_channels=10,
                       data_dir=directory,
                       test_start_date=None,
                       data_frac=1)

if __name__ == '__main__':
    main()
