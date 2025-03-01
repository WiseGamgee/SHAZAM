import os
import csv
import yaml
import torch
import pickle
from tqdm import tqdm
from shazam.anomaly_scoring import SDIM
from shazam.model import ConditionalUNet
from shazam.helpers import load_img_from_file, date_to_polar_period, normalise_image_tensor, save_tensor_as_image


def main():
    # Import test configuration file from local directory
    with open("0_config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    # Get model parameters from configuration file
    model_params = config["MODEL_PARAMS"]

    # Get results folder name using key params
    pfx = config["RESULTS_DIR_PREFIX"]
    tp = config["TIME_PERIOD"]
    lr = model_params["learning_rate"]
    lf = model_params["loss_function"]
    norm = config["NORMALISE"]

    # Create local results directory if folder is missing
    results_dir = f"results/{pfx}_{tp}_{lr}_{lf}/validate"
    if norm:
        results_dir = f"results/{pfx}_{tp}_{lr}_{lf}_norm/validate"
    if os.path.exists(results_dir) is False:
        os.makedirs(results_dir)

    # Save hyperparams as file
    config_dir = f"{results_dir}/configuration_file.yml"
    with open(config_dir, 'w') as hp_file:
        yaml.dump(config, hp_file, default_flow_style=False)

    # Set precision
    torch.set_float32_matmul_precision(config["FLOAT_PRECISION"])

    #
    checkpoint = config["TRAIN_TEST_PARAMS"]["validation_checkpoint"]
    if checkpoint:
        # Load existing model if checkpoint exists
        model = ConditionalUNet.load_from_checkpoint(checkpoint,
                                                     img_channels=config["IMG_CHANNELS"],
                                                     hidden_channels=model_params["hidden_channels"],
                                                     conditional_variables=4,
                                                     kernel_size=model_params["kernel_size"],
                                                     learning_rate=model_params["learning_rate"],
                                                     batch_norm=model_params["batch_norm"],
                                                     upsample_mode=model_params["upsample_mode"],
                                                     loss_function=model_params["loss_function"],
                                                     results_dir=f"{results_dir}")

    else:
        # Load existing model from default checkpoint path
        model = ConditionalUNet.load_from_checkpoint(f"results/{pfx}_{tp}_{lr}_{lf}_norm/trained_model.ckpt",
                                                     img_channels=config["IMG_CHANNELS"],
                                                     hidden_channels=model_params["hidden_channels"],
                                                     conditional_variables=4,
                                                     kernel_size=model_params["kernel_size"],
                                                     learning_rate=model_params["learning_rate"],
                                                     batch_norm=model_params["batch_norm"],
                                                     upsample_mode=model_params["upsample_mode"],
                                                     loss_function=model_params["loss_function"],
                                                     results_dir=f"{results_dir}")

    # Load mean image
    mean_dir = config["MEAN_PATCH_DIR"]
    mean_dirs = [f for f in os.listdir(mean_dir) if f.endswith('.npy')]
    mean_img = torch.zeros(size=(config["IMG_CHANNELS"], config["IMG_SIZE"], config["IMG_SIZE"]))
    for patch_name in mean_dirs:
        # Load mean patch from file and convert to torch
        row, col = map(int, patch_name.split(".npy")[0].split('_')[-2:])
        mu, _ = load_img_from_file(img_folder=config["MEAN_PATCH_DIR"],
                                   img_name=patch_name,
                                   get_date=False)

        # Insert patch into overall mean image
        x_0, x_1 = col * config["PATCH_SIZE"], (col + 1) * config["PATCH_SIZE"]
        y_0, y_1 = row * config["PATCH_SIZE"], (row + 1) * config["PATCH_SIZE"]
        mean_img[..., y_0:y_1, x_0:x_1] = mu

    # Normalise mean image
    if config["NORMALISE"]:
        mean_img = normalise_image_tensor(mean_img,
                                          channels_min=config["NORM_MIN"],
                                          channels_max=config["NORM_MAX"])

    # Loop through full images for anomaly detection
    data_dir = config["TRAIN_IMG_DIR"]
    img_dirs = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    mae_dict = {}
    sdim_dict = {}
    sdim_calculator = SDIM()
    for img_name in tqdm(img_dirs):
        # Load image from file and convert to torch
        img, date = load_img_from_file(img_folder=data_dir,
                                       img_name=img_name)
        t = date_to_polar_period(date_str=date, time_period=config["TIME_PERIOD"])

        if config["NORMALISE"]:
            img = normalise_image_tensor(img,
                                         channels_min=config["NORM_MIN"],
                                         channels_max=config["NORM_MAX"])

        # Apply model to entire image of region
        x_hat = model.forward_img(mean_img=mean_img,
                                  t=t,
                                  img=img,
                                  patch_size=config["PATCH_SIZE"])

        # Get MAE
        mae = torch.nn.functional.l1_loss(img[None], x_hat[None])
        mae_dict[date] = round(mae.item(), 4)

        # Get SSIM and similarity map
        sdim_score, sdim_map = sdim_calculator.forward(img, x_hat)
        sdim_dict[date] = sdim_score

        # Save input, predicted and difference map images
        if config["SAVE_IMGS"]:
            file_name = img_name.rsplit('.', 1)[0]

            # Save predicted image
            save_tensor_as_image(filename=f"{results_dir}/{file_name}_pred.png",
                                 tensor=x_hat,
                                 rgb_bands=config["RGB_BANDS"])

            # Save input image
            save_tensor_as_image(filename=f"{results_dir}/{file_name}_true.png",
                                 tensor=img,
                                 rgb_bands=config["RGB_BANDS"])

            # Save anomaly heatmap
            sdim_score = str(round(sdim_score, 4))
            save_tensor_as_image(filename=f"{results_dir}/{file_name}_dmap_{sdim_score}.png",
                                 tensor=sdim_map,
                                 rgb_bands=None,
                                 v_min=0,
                                 v_max=1,
                                 normalise=False)

    # Save sdim values for training data
    with open(f"{results_dir}/sdim_validation_dict", 'wb') as file:
        pickle.dump(sdim_dict, file)

    # Convert dictionary values to tensors
    mae_values = torch.tensor(list(mae_dict.values()))
    sdim_values = torch.tensor(list(sdim_dict.values()))

    # Calculate mean and standard deviation for mae_dict
    mae_mean = torch.mean(mae_values)
    mae_std = torch.std(mae_values)

    # Calculate mean and standard deviation for sdim_dict
    sdim_mean = torch.mean(sdim_values)
    sdim_std = torch.std(sdim_values)

    # Write the results to a CSV file
    with open(f'{results_dir}/metrics.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Metric', 'Mean', 'Standard Deviation'])
        writer.writerow(['MAE', mae_mean.item(), mae_std.item()])
        writer.writerow(['SDIM', sdim_mean.item(), sdim_std.item()])


if __name__ == '__main__':
    main()
