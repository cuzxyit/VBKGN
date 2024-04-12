import models.arch_util as arch_util
import torch.nn as nn
from models.backbones.resnet import ResnetBlock
from models.backbones.skip.skip import skip
import torch.nn.init as init

class KernelDIP(nn.Module):
    """
    DIP (Deep Image Prior) for blur kernel
    """

    def __init__(self, opt):
        super(KernelDIP, self).__init__()

        norm_layer = arch_util.get_norm_layer("none")
        n_blocks = opt["n_blocks"]
        nf = opt["nf"]
        padding_type = opt["padding_type"]
        use_dropout = opt["use_dropout"]
        kernel_dim = opt["kernel_dim"]

        input_nc = 64
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, nf, kernel_size=7, padding=0, bias=True),
            norm_layer(nf),
            nn.ReLU(True),
        ]

        n_downsampling = 5
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            input_nc = min(nf * mult, kernel_dim)
            output_nc = min(nf * mult * 2, kernel_dim)
            model += [
                nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=2, padding=1, bias=True),
                norm_layer(nf * mult * 2),
                nn.ReLU(True),
            ]

        for i in range(n_blocks):  # add ResNet blocks
            model += [
                ResnetBlock(
                    kernel_dim,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=True,
                )
            ]

        self.model = nn.Sequential(*model)

    def forward(self, noise):
        return self.model(noise)

class SRCNN(nn.Module):
    def __init__(self, upscale_factor):
        super(SRCNN, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)
class ImageDIP(nn.Module):
    """
    DIP (Deep Image Prior) for sharp image
    """

    def __init__(self, opt):
        super(ImageDIP, self).__init__()

        input_nc = opt["input_nc"]
        output_nc = opt["output_nc"]

        self.model = skip(
            input_nc,
            output_nc,
            num_channels_down=[128, 128, 128, 128, 128],
            num_channels_up=[128, 128, 128, 128, 128],
            num_channels_skip=[16, 16, 16, 16, 16],
            upsample_mode="bilinear",
            need_sigmoid=True,
            need_bias=True,
            pad=opt["padding_type"],
            act_fun="LeakyReLU",
        )

    def forward(self, img):

        return self.model(img)
