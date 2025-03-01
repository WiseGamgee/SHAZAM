import os
import torch
import random
import pytorch_lightning as pl
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from shazam.helpers import load_img_from_file, date_to_polar_period, patch_name_to_coords, normalise_image_tensor


class PatchDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 mean_dir: str,
                 patch_size: int,
                 img_size: int,
                 normalise: bool = False,
                 norm_min: list = None,
                 norm_max: list = None,
                 time_period: str = "month",
                 augment=False):
        # Store inputs as attributes
        self.data_dir = data_dir
        self.mean_dir = mean_dir
        self.patch_size = patch_size
        self.normalise = normalise
        self.augment = augment
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.time_period = time_period

        # Get number of patches along each image dimension
        self.patches_per_dim = img_size // patch_size

        # Get list of all patches in data directory
        self.patch_dirs = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

        # Get list of all patches in mean directory
        mean_dirs = [f for f in os.listdir(mean_dir) if f.endswith('.npy')]
        self.mean_dict = {}
        for patch in mean_dirs:
            parts = patch.split(".npy")[0].rsplit('_')[-2:]
            row, col = int(parts[0]), int(parts[1])
            self.mean_dict[(row, col)] = patch

        # Calculate total number of patches within the dataset
        self.num_patches = len(self.patch_dirs)

    def __len__(self):
        return self.num_patches

    def transform(self, mu, x):
        # Random horizontal flipping
        if random.random() > 0.5:
            mu = TF.hflip(mu)
            x = TF.hflip(x)

        # Random vertical flipping
        if random.random() > 0.5:
            mu = TF.vflip(mu)
            x = TF.vflip(x)
        return mu, x

    def __getitem__(self, idx):
        # Get image name
        patch_name = self.patch_dirs[idx]

        # Load image from file and convert to torch
        x, date = load_img_from_file(img_folder=self.data_dir,
                                     img_name=patch_name)

        # Get time-series polar coordinates from date
        t = date_to_polar_period(date_str=date,
                                 time_period=self.time_period)

        # Get coordinate information for current patch
        p = patch_name_to_coords(patch_name, patches_per_dim=self.patches_per_dim)

        # Load mean image from file and convert to torch
        row, col = map(int, patch_name.split(".npy")[0].split('_')[-2:])
        mu, _ = load_img_from_file(img_folder=self.mean_dir,
                                   img_name=self.mean_dict[(row, col)],
                                   get_date=False)

        # Transform data (if specified)
        if self.augment:
            mu, x = self.transform(mu, x)

        # Normalise input imagery
        if self.normalise:
            # Check if min-max values for normalisation are provided
            if self.norm_min is not None:
                x = normalise_image_tensor(img=x,
                                           channels_min=self.norm_min,
                                           channels_max=self.norm_max)
                mu = normalise_image_tensor(img=mu,
                                            channels_min=self.norm_min,
                                            channels_max=self.norm_max)

            # Relative normalisation with inputs' own min-max
            else:
                # Get min and max values of inputs
                x_min, x_max = torch.min(x), torch.max(x)

                # Scale input and output
                range_ = x_max - x_min
                x = (x - x_min) / range_

        # Set label as equal to 0 (only normal data in training set)
        label = torch.Tensor([0]).float()

        # Return input and output for current sample
        return mu, t, p, x, label


class SITSDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_dir: str,
                 mean_dir: str,
                 test_dir: str,
                 patch_size: int,
                 img_size: int,
                 time_period: str = "daily",
                 val_dir: str = None,
                 batch_size: int = 1,
                 num_cpus: int = 1,
                 normalise: bool = False,
                 norm_min: list = None,
                 norm_max: list = None,
                 augment=False):
        super().__init__()

        # Store inputs as attributes
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_cpus = num_cpus
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.normalise = normalise
        self.batch_size = batch_size
        self.augment = augment
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.mean_dir = mean_dir
        self.train_dir = train_dir
        self.time_period = time_period

        # Create dataset attributes
        self.val_dataset = None
        self.test_dataset = None
        self.train_dataset = None

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        # Assign training dataset for use in dataloaders
        if stage == "fit":
            self.train_dataset = PatchDataset(data_dir=self.train_dir,
                                              mean_dir=self.mean_dir,
                                              patch_size=self.patch_size,
                                              img_size=self.img_size,
                                              time_period=self.time_period,
                                              normalise=self.normalise,
                                              norm_min=self.norm_min,
                                              norm_max=self.norm_max,
                                              augment=self.augment)

            if self.val_dir:
                self.val_dataset = PatchDataset(data_dir=self.val_dir,
                                                mean_dir=self.mean_dir,
                                                patch_size=self.patch_size,
                                                img_size=self.img_size,
                                                time_period=self.time_period,
                                                normalise=self.normalise,
                                                norm_min=self.norm_min,
                                                norm_max=self.norm_max,
                                                augment=self.augment)

        # Assign test dataset for use in dataloader
        if stage == "test":
            self.test_dataset = PatchDataset(data_dir=self.test_dir,
                                             mean_dir=self.mean_dir,
                                             patch_size=self.patch_size,
                                             img_size=self.img_size,
                                             time_period=self.time_period,
                                             normalise=self.normalise,
                                             norm_min=self.norm_min,
                                             norm_max=self.norm_max,
                                             augment=self.augment)

        # if stage == "predict":
        # pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_cpus,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_cpus,
                          persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_cpus)


# test = PatchDataset(data_dir="../data/gc/train_patched/",
#                     mean_dir="../data/gc/mean_patched/",
#                     patch_size=32,
#                     time_period="month",
#                     augment=True,
#                     )
#
#
# mu, t, x, label = test.__getitem__(0)
#
# print(t)
# print(x.shape)
#
# import matplotlib.pyplot as plt
# #plt.imshow(x.detach().numpy().T * 3.5)
# plt.imshow(x[[2,1,0], ...].detach().numpy().T * 3.5)
# plt.show()
#
# plt.imshow(mu[[2,1,0], ...].detach().numpy().T * 3.5)
# plt.show()
#
#
# # Example usage:
# # img_tensor is a tensor of shape (c, h, w)
# # patch_size is the size of the patches (e.g., 16)
# patches = image_to_patches(x, 64)
# x_rec = depatchify(patches, img_size=1024, img_channels=12)
# print(patches.shape)  # Should output (256, 3, 16, 16)
#
# plt.imshow(x[[3,2,1], ...].detach().numpy().T * 3.5)
# plt.show()
#print(test.label_dict)
#
# test = SITSDataModule(train_dir="../data/gc/train_patched",
#                       val_dir="../data/gc/val_patched",
#                       test_dir="../data/gc/test_patched",
#                       mean_img_dir="../data/gc/mean_img.npy",
#                       test_label_dir="bazinga",
#                       img_size=32,
#                       batch_size=1,
#                       num_cpus=1,
#                       normalise=False)
#
# test.prepare_data()
#
# test.setup("fit")
# #
# x, mu, g, label = test.train_dataset.__getitem__(1)
# #
# import matplotlib.pyplot as plt
# print(x.shape, mu.shape, g.shape)
# print(label)
# plt.figure()
# plt.imsave("test.png", x[[2,1,0], ...].detach().numpy().T * 3.5)
# plt.imsave("mu.png", mu[[2,1,0], ...].detach().numpy().T * 3.5)

# test = calculate_mean_image(data_dir="../data/gc/train_patched",
#                             img_channels=12,
#                             num_patches=32,
#                             patch_size=32)
#
# print(test.shape)
# import matplotlib.pyplot as plt
# plt.imshow(test[0, 0][ ..., [3,2,1]] * 3.5)
# plt.show()