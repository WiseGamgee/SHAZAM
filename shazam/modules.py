import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 batch_norm: bool = False):
        """
        Block of layers that downsamples the input into a deeper feature-space
        :param in_channels: number of channels in input
        :param out_channels: number of channels in output
        :param kernel_size: size of filter for convolutional layers
        :param batch_norm: boolean indicating whether to use batch normalisation or not
        """
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               padding=(kernel_size - 1) // 2,
                               stride=1)

        if batch_norm:
            self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        else:
            self.bn1 = None

        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               padding=1,
                               stride=1)

        if batch_norm:
            self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        else:
            self.bn2 = None

        self.act2 = nn.GELU()

    def forward(self, x):
        # First convolutional block
        x = self.conv1(x)
        if self.bn1:
            x = self.bn1(x)
        x = self.act1(x)

        # Second convolutional block
        x = self.conv2(x)
        if self.bn2:
            x = self.bn2(x)
        x = self.act2(x)
        return x


class DownBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 batch_norm: bool = False):
        """
        Block of layers that downsamples the input into a deeper feature-space
        :param in_channels: number of channels in input
        :param out_channels: number of channels in output
        :param kernel_size: size of filter for convolutional layers
        :param batch_norm: boolean indicating whether to use batch normalisation or not
        """
        super().__init__()

        # Instantiate max pooling layer
        self.max_pool = nn.MaxPool2d(kernel_size=2,
                                     stride=2,
                                     padding=0)

        # Instantiate double convolution block
        self.double_conv = DoubleConv(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      batch_norm=batch_norm)

    def forward(self, x):
        # Down sample input
        x = self.max_pool(x)
        return self.double_conv(x)


class Encoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: list,
                 kernel_size: int = 3,
                 batch_norm: bool = False):
        """
        Encodes input image into a vector.
        :param in_channels: number of input channels
        :param hidden_channels: list of arch. channels for creating encoder blocks (e.g. [8, 16, 32])
        :param kernel_size: size of filter for all convolutional layers
        """
        super().__init__()

        # Initialise network for taking image input
        self.blocks = nn.ModuleList()
        self.blocks.append(DoubleConv(in_channels=in_channels,
                                      out_channels=hidden_channels[0],
                                      kernel_size=kernel_size,
                                      batch_norm=batch_norm))

        # Build encoder network
        for i in range(len(hidden_channels) - 1):
            block = DownBlock(in_channels=hidden_channels[i],
                              out_channels=hidden_channels[i + 1],
                              kernel_size=kernel_size,
                              batch_norm=batch_norm)
            self.blocks.append(block)

    def forward(self, x):
        # Loop through all blocks in encoder
        residuals = []
        for block in self.blocks:
            x = block(x)

            # Store convolutional residuals
            residuals.append(x)
        # Return encoder output and residuals (last block excluded as residual not needed)
        return x, residuals[:-1]


class UpBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 conditional_variables: int,
                 kernel_size: int = 3,
                 batch_norm: bool = False,
                 upsample_mode: str = "bilinear"):
        """
        Block of layers that upsamples the input into a more shallow feature-space
        :param in_channels: number of channels in input
        :param out_channels: number of channels in output
        :param kernel_size: size of filter for convolutional layers
        :param batch_norm: boolean indicating whether to use batch normalisation or not
        :param upsample_mode: type of upsampling to be used (e.g. bilinear, nearest neighbours, etc.)
        """
        super().__init__()
        # Define upsampling layer
        self.upsample = nn.Upsample(scale_factor=2,
                                    mode=upsample_mode)

        # Define additional convolution layer for halving channel size
        self.up_conv = nn.Conv2d(in_channels=in_channels,
                                 out_channels=in_channels // 2,
                                 kernel_size=2,
                                 stride=1,
                                 padding=1)

        # Define double conv
        self.double_conv = DoubleConv(in_channels=in_channels + conditional_variables,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      batch_norm=batch_norm)

    def forward(self, x, residual):
        # Upsample features
        x = self.up_conv(self.upsample(x))

        # Pad input (downsampling can result in dimension losses when upsampling again)
        width_diff = residual.shape[3] - x.shape[3]
        height_diff = residual.shape[2] - x.shape[2]
        x = F.pad(input=x,
                  pad=(width_diff // 2,
                       width_diff - width_diff // 2,
                       height_diff // 2,
                       height_diff - height_diff // 2))

        # Concatenate residuals onto features
        x = torch.cat([x, residual], dim=1)
        return self.double_conv(x)


class Decoder(nn.Module):
    def __init__(self,
                 hidden_channels: list,
                 out_channels: int,
                 conditional_variables: int,
                 kernel_size: int = 3,
                 batch_norm: bool = False,
                 upsample_mode: str = "bilinear"):
        """
        Decodes vector of latent feature into output.
        :param hidden_channels: list of arch. channels for creating decoder blocks (e.g. [32, 16, 8])
        :param kernel_size: size of filter for all convolutional layers
        """
        super().__init__()

        # Create initial convolutional block
        self.blocks = nn.ModuleList()

        # Build network
        for i in range(len(hidden_channels) - 1):
            block = UpBlock(in_channels=hidden_channels[i],
                            out_channels=hidden_channels[i + 1],
                            conditional_variables=conditional_variables,
                            kernel_size=kernel_size,
                            batch_norm=batch_norm,
                            upsample_mode=upsample_mode)
            self.blocks.append(block)

        #
        # self.double_conv = DoubleConv(in_channels=hidden_channels[-1],
        #                               out_channels=hidden_channels[-1],
        #                               kernel_size=kernel_size,
        #                               batch_norm=batch_norm)

        #
        self.output = nn.Conv2d(in_channels=hidden_channels[-1],
                                out_channels=out_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0)

    def forward(self, x, residuals):
        # Loop through decoder network
        for block, residual in zip(self.blocks, reversed(residuals)):
            # Calculate output of decoder block
            x = block(x, residual)
        #x = self.double_conv(x)
        return self.output(x)


# img_channels = 10
# conditional_variables = 0
# hidden_channels = [32, 64, 128]
# x = torch.randn(size=(3, img_channels+conditional_variables, 32, 32))
# encoder = Encoder(in_channels=img_channels+conditional_variables,
#                   hidden_channels=hidden_channels)
#
# x, res = encoder(x)
#
# decoder = Decoder(hidden_channels=hidden_channels[::-1],
#                   out_channels=img_channels,
#                   conditional_variables=conditional_variables)
# print(decoder(x, res).shape)
