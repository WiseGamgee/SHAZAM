import os
import csv
import yaml
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from shazam.model import ConditionalUNet
from shazam.dataloading import SITSDataModule


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
    results_dir = f"results/{pfx}_{tp}_{lr}_{lf}"
    if norm:
        results_dir = f"results/{pfx}_{tp}_{lr}_{lf}_norm"
    if os.path.exists(results_dir) is False:
        os.makedirs(results_dir)

    # Save hyperparams as file
    config_dir = f"{results_dir}/configuration_file.yml"
    with open(config_dir, 'w') as hp_file:
        yaml.dump(config, hp_file, default_flow_style=False)

    # Set precision
    torch.set_float32_matmul_precision(config["FLOAT_PRECISION"])

    # Create datamodule
    sits_dm = SITSDataModule(train_dir=config["TRAIN_PATCH_DIR"],
                             val_dir=config["VAL_PATCH_DIR"],
                             test_dir=config["TEST_PATCH_DIR"],
                             mean_dir=config["MEAN_PATCH_DIR"],
                             patch_size=config["PATCH_SIZE"],
                             img_size=config["IMG_SIZE"],
                             time_period=config["TIME_PERIOD"],
                             batch_size=config["BATCH_SIZE"],
                             num_cpus=config["N_CPUS"],
                             normalise=config["NORMALISE"],
                             norm_min=config["NORM_MIN"],
                             norm_max=config["NORM_MAX"],
                             augment=config["AUGMENT"])

    #
    checkpoint = config["TRAIN_TEST_PARAMS"]["training_checkpoint"]
    if checkpoint:
        # Load existing model
        model = ConditionalUNet.load_from_checkpoint(checkpoint,
                                                     img_channels=config["IMG_CHANNELS"],
                                                     hidden_channels=model_params["hidden_channels"],
                                                     conditional_variables=4,
                                                     kernel_size=model_params["kernel_size"],
                                                     learning_rate=model_params["learning_rate"],
                                                     batch_norm=model_params["batch_norm"],
                                                     upsample_mode=model_params["upsample_mode"],
                                                     loss_function=model_params["loss_function"],
                                                     results_dir=f"{results_dir}/train")

    else:
        # Instantiate empty model
        model = ConditionalUNet(img_channels=config["IMG_CHANNELS"],
                                hidden_channels=model_params["hidden_channels"],
                                conditional_variables=4,
                                kernel_size=model_params["kernel_size"],
                                learning_rate=model_params["learning_rate"],
                                batch_norm=model_params["batch_norm"],
                                upsample_mode=model_params["upsample_mode"],
                                loss_function=model_params["loss_function"],
                                results_dir=f"{results_dir}/train")

    # Train model
    n_epochs = config["N_EPOCHS"]
    trainer = pl.Trainer(reload_dataloaders_every_n_epochs=1,
                         accelerator=config["ACCELERATOR"],
                         devices=config["N_GPUS"],
                         max_epochs=n_epochs)
    print("Training...")
    trainer.fit(model, sits_dm)

    # Save model
    trainer.save_checkpoint(f"{results_dir}/trained_model.ckpt")

    # Print final loss
    final_loss = model.custom_logger["train_loss"][-1]
    print(f"The final loss is: {final_loss}")

    # Save losses in csv file
    loss_dir = f"{results_dir}/train/training_loss.csv"
    with open(loss_dir, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows([[loss] for loss in model.custom_logger["train_loss"]])

    # Plot training loss results
    plt.figure(figsize=(25, 10))
    plt.xticks(rotation=90)
    plt.plot(model.custom_logger["train_loss"])
    plt.xlabel("No. of Samples")
    plt.ylabel("Loss")
    plt.title(f"SIUNet Training Loss Function ({n_epochs} epochs).")
    plt.savefig(f"{results_dir}/train/training_loss.png", format="png")

    # Plot validation loss results
    plt.figure(figsize=(25, 10))
    plt.xticks(rotation=90)
    plt.plot(model.custom_logger["val_loss"])
    plt.xlabel("No. of Samples")
    plt.ylabel("Loss")
    plt.title(f"SIUNet Validation Loss Function ({n_epochs} epochs).")
    plt.savefig(f"{results_dir}/train/validation_loss.png", format="png")


if __name__ == '__main__':
    main()
