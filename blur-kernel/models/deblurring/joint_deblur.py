import torch
import numpy as np
import utils.util as util
from models.deblurring.image_deblur import ImageDeblur
from tqdm import tqdm
from models.deblurring.utils.reblur import forward_reblur, apply_saturation_function
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from PIL import Image

class JointDeblur(ImageDeblur):
    def __init__(self, opt):
        super(JointDeblur, self).__init__(opt)

    def deblur(self, y):
        """Deblur image
        Args:
            y: Blur image

        """

        # def tensor2im(img, imtype=np.uint8):
        #     image_numpy = img.cpu().float().numpy()
        #     image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 0.5) * 255.0
        #     return image_numpy.astype(imtype)
        # y = util.img2tensor(y).unsqueeze(0).cuda()

        # y = y ** 2.2
        # a=self.SR_block(y)
        self.prepare_DIPs()
        self.reset_optimizers()

        warmup_k = torch.load(self.opt["warmup_k_path"]).cuda()
        self.warmup(y, warmup_k)

        # Input vector of DIPs is sampled from N(z, I)

        print("Deblurring")
        reg_noise_std = self.opt["reg_noise_std"]
        for step in tqdm(range(self.opt["num_iters"])):
            dip_zx_rand = self.dip_zx + reg_noise_std * torch.randn_like(self.dip_zx).cuda()
            dip_zk_rand = self.dip_zk + reg_noise_std * torch.randn_like(self.dip_zk).cuda()

            self.x_optimizer.zero_grad()
            self.k_optimizer.zero_grad()

            self.x_scheduler.step()
            self.k_scheduler.step()

            x = self.x_dip(dip_zx_rand)

            k = self.k_dip(dip_zk_rand)

            # y_k=y** 2.2-0.5

            self.kernels, mask, mu_z, logvar_z, _ = self.kernel_wizard(y)

            x = x ** 2.2

            fake_y = forward_reblur(x, self.kernels, mask, size='same', padding_mode='reflect',
                                        manage_saturated_pixels=False, max_value=1)

            # fake_y=fake_yk** (1.0 / 2.2)
            # forward_reblur(output_ph, kernels, masks,  size='same',
            #                padding_mode='reflect', manage_saturated_pixels=False, max_value=1)
            # fake_y = self.kernel_wizard.adaptKernel(x, k)

            if step < self.opt["num_iters"] // 2:
                total_loss = 6e-1 * self.perceptual_loss(fake_y, y)
                total_loss += 1 - self.ssim_loss(fake_y, y)
                total_loss += 5e-5 * torch.norm(self.kernels)
                total_loss += 2e-2 * self.laplace_penalty(x)
            else:
                total_loss = self.perceptual_loss(fake_y, y)
                total_loss += 5e-2 * self.laplace_penalty(x)
                total_loss += 5e-4 * torch.norm(self.kernels)

            total_loss.backward()

            self.x_optimizer.step()
            self.k_optimizer.step()


            # debugging
            # if step % 100 == 0:
            #     print(torch.norm(k))
            #     print(f"{self.k_optimizer.param_groups[0]['lr']:.3e}")

        # j=self.SR_block(x)

        # j=x.detach()

        # output_img = tensor2im(torch.clamp(j[0], 0, 1) - 0.5)
        # save_image(output_img, os.path.join(args.output_folder, img_name + '_restored.png'))
        # return util.tensor2img(x[0] - 0.5.detach())

        return util.tensor2img(x.detach()),self.kernels
        # return  output_img
