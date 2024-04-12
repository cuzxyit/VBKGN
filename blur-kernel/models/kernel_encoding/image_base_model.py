import logging
from collections import OrderedDict

import models.lr_scheduler as lr_scheduler
import torch
import torch.nn as nn
import os
from models.kernel_encoding.base_model import BaseModel
from models.kernel_encoding.kernel_wizard import KernelWizard
from models.losses.charbonnier_loss import CharbonnierLoss
from torch.nn.parallel import DataParallel, DistributedDataParallel
from math import log,ceil
from torch.autograd import Variable
from skimage.io import imread, imsave
from skimage import img_as_ubyte
from torch.optim.lr_scheduler import StepLR
import torchvision.utils as vutils
from models.losses.ssim_loss import SSIM
import numpy as np
from models.kernel_encoding.TwoHeadsNetwork import TwoHeadsNetwork
import cv2

from models.deblurring.utils.reblur import forward_reblur,compute_gradient_loss

logger = logging.getLogger("base")


class ImageBaseModel(BaseModel):
    def __init__(self, opt):
        super(ImageBaseModel, self).__init__(opt)

        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt["train"]

        # define network and load pretrained models

        self.netG= TwoHeadsNetwork(K=opt["EDnet"]["K"],nz=opt["EDnet"]["nz"]).to(self.device)

        self.use_vae = opt["KernelWizard"]["use_vae"]
        if opt["dist"]:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()
            #self.netD.train()
            # self.netM.train()
           # self.netD.train()

            # loss
            loss_type = train_opt["pixel_criterion"]
            if loss_type == "l1":
                self.cri_pix = nn.L1Loss(reduction="sum").to(self.device)
            elif loss_type == "l2":
                self.cri_pix = nn.MSELoss(reduction="sum").to(self.device)
            elif loss_type == "cb":
                self.cri_pix = CharbonnierLoss().to(self.device)
            else:
                raise NotImplementedError(
                    "Loss type [{:s}] is not\
                                          recognized.".format(
                        loss_type
                    )
                )
            self.l_pix_w = train_opt["pixel_weight"]
            self.l_kl_w = train_opt["kl_weight"]

            # optimizers
            wd_G = train_opt["weight_decay_G"] if train_opt["weight_decay_G"] else 0
            params = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning(
                            "Params [{:s}] will not\
                                       optimize.".format(
                                k
                            )
                        )
            optim_params = [
                {"params": params, "lr": train_opt["lr_G"]},
            ]

            self.optimizer_G = torch.optim.Adam(
                optim_params, lr=train_opt["lr_G"], weight_decay=wd_G, betas=(train_opt["beta1"], train_opt["beta2"])
            )
            # self.x_optimizer = torch.optim.Adam(self.netM.parameters(), lr=train_opt["x_lr"])
            self.optimizers.append(self.optimizer_G)
           # self.optimizerD = torch.optim.Adam(self.netD.parameters(), lr=train_opt["lrD"])
            #self.schedulerD = torch.optim.lr_scheduler.MultiStepLR(self.optimizerD, train_opt["restarts"], gamma=0.5)
            # self.x_scheduler = StepLR(self.x_optimizer, step_size=train_opt["niter"] // 5, gamma=0.7)
            # schedulers
            if train_opt["lr_scheme"] == "MultiStepLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(
                            optimizer,
                            train_opt["lr_steps"],
                            # restarts=train_opt["restarts"],
                            # weights=train_opt["restart_weights"],
                            gamma=train_opt["lr_gamma"],
                            clear_state=train_opt["clear_state"],
                        )
                    )
            elif train_opt["lr_scheme"] == "CosineAnnealingLR_Restart":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer,
                            train_opt["T_period"],
                            eta_min=train_opt["eta_min"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                        )
                    )
            else:
                raise NotImplementedError()

            self.log_dict = OrderedDict()


    def reparametrize(self,mu_k,log_k):
        std = log_k.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu_k + std * eps

    def feed_data(self, data, need_GT=True):
        self.LQ = data["LQ"].to(self.device)
        self.HQ = data["HQ"].to(self.device)

    def set_params_lr_zero(self, groups):
        # fix normal module
        for group in groups:
            self.optimizers[0].param_groups[group]["lr"] = 0

    def optimize_parameters(self, step):
        batchsz, _, _, _ = self.LQ.shape
        log_max = log(1e4)
        log_min = log(1e-8)
        self.mse = nn.MSELoss().to(self.device)
        self.ssim = SSIM().to(self.device)
        self.optimizer_G.zero_grad()

        blurry_tensor_to_compute_kernels = self.LQ ** 2.2-0.5
        self.kernels, mask, mu_z, logvar_z, _ = self.netG( blurry_tensor_to_compute_kernels)

        sharp_hp=self.HQ** 2.2
        output_reblurred = forward_reblur(sharp_hp, self.kernels, mask, size='same', padding_mode='reflect',
                                          manage_saturated_pixels=True, max_value=1)
        output_reblurred=output_reblurred** (1.0 / 2.2)
        logvar_z.clamp_(min=log_min, max=log_max)  # clip
        var_z = torch.exp(logvar_z)
        kl_gauss_z = 0.5 * torch.mean(mu_z ** 2 + (var_z - 1 - logvar_z))

        g_loss=compute_gradient_loss(output_reblurred,self.HQ,self.kernels,mask)

        l_pix = self.l_pix_w * self.cri_pix(output_reblurred , self.LQ)
       #  l_pix=self.ssim (output_reblurred,self.LQ)
        l_total = l_pix+kl_gauss_z+g_loss#g_loss_fake+kl_gauss_z+kl_gauss_blossh++loss_f

        l_total.backward()
        # self.x_optimizer.step()
        self.optimizer_G.step()

       # self.schedulerD.step()
       #  self.x_scheduler.step()

        x1 = vutils.make_grid(self.LQ.data, normalize=True, scale_each=True) #清晰
        x2=vutils.make_grid( output_reblurred.data, normalize=True, scale_each=True)#模糊
        x3 = vutils.make_grid(self.HQ.data, normalize=True, scale_each=True)  # 清晰
        listk = self.kernels.chunk(self.opt["EDnet"]["K"], 1)


        # set log
        self.log_dict["l_pix"] = l_pix.item() / batchsz
        # self.log_dict["l_pixk"] = kl_gauss_z.item() / batchsz
        # self.log_dict["l_pixx"] = kl_gauss_b.item() / batchsz
        self.log_dict["l_total"] = l_total.item() / batchsz
        self.log_dict["huZ"] = x1
        self.log_dict["hu"] = x2
        self.log_dict["qingZ"] = x3
        self.log_dict["HZ"] = listk


    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict



    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["LQ"] = self.LQ.detach()[0].float().cpu()
        out_dict["rlt"] = self.fake_LQ.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = "{} - {}".format(self.netG.__class__.__name__, self.netG.module.__class__.__name__)
        else:
            net_struc_str = "{}".format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info(
                "Network G structure: {}, \
                        with parameters: {:,d}".format(
                    net_struc_str, n
                )
            )
            logger.info(s)

    def load(self):
        if self.opt["path"]["pretrain_model_G"]:
            load_path_G = self.opt["path"]["pretrain_model_G"]
            if load_path_G is not None:
                logger.info(
                    "Loading model for G [{:s}]\
                            ...".format(
                        load_path_G
                    )
                )
                self.load_network(load_path_G, self.netG, self.opt["path"]["strict_load"])

    def save(self, iter_label):
        self.save_network(self.netG, "G", iter_label)


    def save_kernel(self, iter_label):
        listk = self.kernels.chunk(self.opt["EDnet"]["K"], 1)
        for k in range(self.opt["EDnet"]["K"]):
            for ba in range(self.opt["EDnet"]["batch_size"]):
                image_name = os.path.join(self.opt["EDnet"]["savei"],
                                          str(iter_label + 1) + "_" + str(k) + "_" + str(ba) + '_kernels' + '.png')
                vutils.save_image(listk[k][ba], image_name, normalize=True)