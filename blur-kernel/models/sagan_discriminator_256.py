# This is  a demo setting of D for patchsize = 256 x 256
# https://github.com/heykeetae/Self-Attention-GAN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.spectral import SpectralNorm
import numpy as np

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""
    def __init__(self,opt):
        super(Discriminator, self).__init__()
        self.imsize = opt["image_size"]
        batch_size = opt["batch_size"]
        conv_dim = opt["conv_dim"]
        layer1 = []
        layer2 = []
        layer3 = []
        layer31 = []
        layer32 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv2d(3, conv_dim, 4, 2, 1)))     #128*128
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1))) # 64 *64
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))  # 32 *32
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer31.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1))) #16*16
        layer31.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer32.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1))) # 8 * 8
        layer32.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        if self.imsize == 256:
            layer4 = []
            layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))  # 4 *4
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)     # 1* 1
            curr_dim = curr_dim*2
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)
        self.l31 = nn.Sequential(*layer31)
        self.l32 = nn.Sequential(*layer32)
        last.append(nn.Conv2d(curr_dim, 1, 4))
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn(1024, 'relu')
        self.attn2 = Self_Attn(2048, 'relu')

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l31(out)
        out = self.l32(out)
        out,p1 = self.attn1(out)
        out=self.l4(out)
        out,p2 = self.attn2(out)
        out=self.last(out)

        return out.squeeze(), p1, p2
