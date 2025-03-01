import os
import torch
import numpy as np
from torch import nn, optim
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from shazam.modules import Encoder, Decoder
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.image import StructuralSimilarityIndexMeasure


class ConditionalUNet(pl.LightningModule):
    def __init__(self,
                 img_channels: int,
                 hidden_channels: list,
                 conditional_variables: int,
                 kernel_size: int = 3,
                 learning_rate: float = 1e-4,
                 batch_norm: bool = False,
                 upsample_mode: str = "bilinear",
                 loss_function: str = "MAE",
                 results_dir: str = "results"):
        """
        :param img_channels:
        :param hidden_channels:
        :param kernel_size:
        :param batch_norm:
        :param upsample_mode:
        """
        super().__init__()
        self.results_dir = results_dir
        self.loss_function = loss_function
        self.learning_rate = float(learning_rate)
        self.save_hyperparameters()

        # Instantiate encoder
        self.encoder = Encoder(in_channels=img_channels + conditional_variables,
                               hidden_channels=hidden_channels,
                               kernel_size=kernel_size,
                               batch_norm=batch_norm)

        # Instantiate decoder
        self.decoder = Decoder(hidden_channels=hidden_channels[::-1],
                               out_channels=img_channels,
                               conditional_variables=conditional_variables,
                               kernel_size=kernel_size,
                               batch_norm=batch_norm,
                               upsample_mode=upsample_mode)

        # Define loss function
        if loss_function == "MAE":
            self.loss_fn = nn.L1Loss()
        elif loss_function == "MSE":
            self.loss_fn = nn.MSELoss()
        elif loss_function == "SDIM":
            self.loss_fn = SDIMLoss()
        elif loss_function == "DUAL":
            self.loss_fn = DualLoss()

        # Metric tracking
        self.epoch_no = 0
        self.custom_logger = {"train_loss": [],
                              "val_loss": [],
                              "test_loss": []}

    # Basic model throughput to get predicted patch
    def model(self, mu, t, p):
        # Convert time and position features into channels and concatenate onto mean image
        b, c, h, w = mu.shape
        mu = torch.cat(tensors=[mu,
                                t.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w),
                                p.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)], dim=1)

        # Run through encoder and bridge
        z, residuals = self.encoder(mu)

        # Concatenate geographic features onto residuals
        for i in range(len(residuals)):
            r = residuals[i]
            b_r, c_r, h_r, w_r = r.shape
            residuals[i] = torch.cat(tensors=[r,
                                              t.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h_r, w_r),
                                              p.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h_r, w_r)], dim=1)

        # Run through model decoder
        x_hat = self.decoder(z, residuals)
        return x_hat

    #
    def forward_img(self, mean_img, t, img, patch_size):
        # Get input dimensions
        c, h, w = img.shape

        # Loop through image and mean patches, creating predicted image and SDIM map
        # Create tensors to store model outputs
        x_hat = torch.zeros(size=(c, h, w))

        # Loop through all patches in input image and mean image
        axis_ratio = h // patch_size
        for i in range(axis_ratio):
            x_0, x_1 = i * patch_size, (i + 1) * patch_size
            for j in range(axis_ratio):
                y_0, y_1 = j * patch_size, (j + 1) * patch_size

                # Get position coords for current patch
                p = (torch.Tensor([j, i]) / (h // patch_size - 1)).float()

                # Get model's predicted patch
                mu_patch = mean_img[..., y_0:y_1, x_0:x_1]
                patch = self.model(mu_patch[None], t[None], p[None])[0]
                x_hat[..., y_0:y_1, x_0:x_1] = patch

        # Return predicted image, SDIM map, and tail structural difference score
        return x_hat

    #
    def sample(self, mu, t, p, device_type="cuda"):
        # Convert time and patch positions into torch tensor
        t_sin = np.sin(2 * np.pi * t / 12)
        t_cos = np.cos(2 * np.pi * t / 12)
        t = torch.Tensor([t_sin, t_cos]).float()

        #
        x_hat = self.model(mu[None], t[None], p[None])
        return x_hat

    # Forward pass for inference
    def forward(self, mu, t, p):
        return self.model(mu, t, p)

    # Training step for a single batch
    def training_step(self, batch, batch_idx):
        # Get batch inputs and true outputs
        mu, t, p, x, _ = batch

        # Get predicted outputs
        x_hat = self.model(mu, t, p)

        # Get batch loss and metrics
        loss = self.loss_fn(input=x_hat, target=x)

        # Logging
        self.log("train_loss", loss.item(), prog_bar=True, on_step=True)
        # for name, param in self.named_parameters():
        #     if param.grad is not None:
        #         grad_mean = param.grad.mean()
        #         self.log(f'grad_{name}', grad_mean, prog_bar=True, on_step=True)
        self.custom_logger["train_loss"].append(loss.item())
        self.custom_logger["train_mean_img"] = mu[0][[2, 1, 0], ...].permute(1, 2, 0).cpu().detach().numpy()
        self.custom_logger["train_real_img"] = x[0][[2, 1, 0], ...].permute(1, 2, 0).cpu().detach().numpy()
        self.custom_logger["train_pred_img"] = x_hat[0][[2, 1, 0], ...].permute(1, 2, 0).cpu().detach().numpy()
        return loss

    def on_train_epoch_end(self):
        # Create results path if doesn't exist
        if os.path.exists(self.results_dir) is False:
            os.makedirs(self.results_dir)

        # Save true image
        real_rgb = self.custom_logger["train_real_img"]
        real_rgb = (real_rgb - np.min(real_rgb)) / (np.max(real_rgb) - np.min(real_rgb))
        plt.imsave(f"{self.results_dir}/train_epoch_{self.epoch_no}_real.png", arr=real_rgb, format="png")
        self.custom_logger["train_real_img"] = None

        # Save mean image
        mean_rgb = self.custom_logger["train_mean_img"]
        mean_rgb = (mean_rgb - np.min(mean_rgb)) / (np.max(mean_rgb) - np.min(mean_rgb))
        plt.imsave(f"{self.results_dir}/train_epoch_{self.epoch_no}_mean.png", arr=mean_rgb, format="png")
        self.custom_logger["train_mean_img"] = None

        # Save most recent input image in sequence
        pred_rgb = self.custom_logger["train_pred_img"]
        pred_rgb = (pred_rgb - np.min(pred_rgb)) / (np.max(pred_rgb) - np.min(pred_rgb))
        plt.imsave(f"{self.results_dir}/train_epoch_{self.epoch_no}_pred.png", arr=pred_rgb, format="png")
        self.custom_logger["train_pred_img"] = None

        # Increment epoch counter by 1
        self.epoch_no += 1

    # Validation step for a single batch
    def validation_step(self, batch, batch_idx):
        # Get batch inputs and true outputs
        mu, t, p, x, _ = batch

        # Get predicted outputs
        x_hat = self.model(mu, t, p)

        # Get batch loss and metrics
        loss = self.loss_fn(input=x_hat, target=x)

        # Logging
        self.log("val_loss", loss.item(), prog_bar=True, on_step=True)
        self.custom_logger["val_loss"].append(loss.item())
        self.custom_logger["val_mean_img"] = mu[0][[2, 1, 0], ...].permute(1, 2, 0).cpu().detach().numpy()
        self.custom_logger["val_real_img"] = x[0][[2, 1, 0], ...].permute(1, 2, 0).cpu().detach().numpy()
        self.custom_logger["val_pred_img"] = x_hat[0][[2, 1, 0], ...].permute(1, 2, 0).cpu().detach().numpy()
        return loss

    def on_validation_epoch_end(self):
        # Create results path if doesn't exist
        if os.path.exists(self.results_dir) is False:
            os.makedirs(self.results_dir)

        # Save true image
        real_rgb = self.custom_logger["val_real_img"]
        real_rgb = (real_rgb - np.min(real_rgb)) / (np.max(real_rgb) - np.min(real_rgb))
        plt.imsave(f"{self.results_dir}/val_epoch_{self.epoch_no}_real.png", arr=real_rgb, format="png")
        self.custom_logger["val_real_img"] = None

        # Save mean image
        mean_rgb = self.custom_logger["val_mean_img"]
        mean_rgb = (mean_rgb - np.min(mean_rgb)) / (np.max(mean_rgb) - np.min(mean_rgb))
        plt.imsave(f"{self.results_dir}/val_epoch_{self.epoch_no}_mean.png", arr=mean_rgb, format="png")
        self.custom_logger["val_mean_img"] = None

        # Save most recent input image in sequence
        pred_rgb = self.custom_logger["val_pred_img"]
        pred_rgb = (pred_rgb - np.min(pred_rgb)) / (np.max(pred_rgb) - np.min(pred_rgb))
        plt.imsave(f"{self.results_dir}/val_epoch_{self.epoch_no}_pred.png", arr=pred_rgb, format="png")
        self.custom_logger["val_pred_img"] = None

    # Test step for a single batch
    def test_step(self, batch, batch_idx):
        # Get batch inputs and true outputs
        mu, t, p, x, label = batch

        # Get predicted outputs
        x_hat = self.model(mu, t, p)

        # Get batch loss and metrics
        loss = self.loss_fn(input=x_hat, target=x)

        # Logging
        self.log("loss", loss.item(), prog_bar=True, on_step=True)
        self.custom_logger["test_loss"].append(loss.item())
        self.custom_logger["test_mean_img"] = mu[0][[2, 1, 0], ...].permute(1, 2, 0).cpu().detach().numpy()
        self.custom_logger["test_real_img"] = x[0][[2, 1, 0], ...].permute(1, 2, 0).cpu().detach().numpy()
        self.custom_logger["test_pred_img"] = x_hat[0][[2, 1, 0], ...].permute(1, 2, 0).cpu().detach().numpy()
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        # return optimiser
        scheduler = ReduceLROnPlateau(optimizer,
                                      mode='min',
                                      factor=0.1,
                                      patience=3,
                                      min_lr=1e-7)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler,
                                 'monitor': 'val_loss'}}

class SDIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure()

    def forward(self, input, target):
        return (1 - self.ssim(input, target)) / 2


class DualLoss(nn.Module):
    def __init__(self, alpha=0.15, kernel_size=11):
        super().__init__()
        self.alpha = alpha
        self.ssim_fn = StructuralSimilarityIndexMeasure(kernel_size=kernel_size)
        self.loss_fn = nn.L1Loss()

    def forward(self, input, target):
        l1_loss = self.loss_fn(input=input, target=target)
        sd_loss = (1 - self.ssim_fn(input, target)) / 2
        return self.alpha * sd_loss + (1 - self.alpha) * l1_loss


# x = torch.randn(size=(3, 10, 32, 32))
# t = torch.ones(size=(3, 2))
# p = torch.ones(size=(3, 2))
# mean_img = torch.mean(x, dim=0)
#
# model = ConditionalUNet(img_channels=10,
#                         hidden_channels=[32, 64, 128],
#                         conditional_variables=4)
#
# print(model.model(x, t, p).shape)




# x_hat, sdim, anom_scores = model.forward_img(mean_img=torch.zeros(size=(3, 10, 256, 256)),
#                                               t=torch.zeros(size=(3, 2)),
#                                               img=torch.randn(size=(3, 10, 256, 256)),
#                                               patch_size=32,
#                                               q=0.95,
#                                               training=True)
#
# import matplotlib.pyplot as plt
# plt.imshow(sdim[0].detach().numpy())
# plt.show()
