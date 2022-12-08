import torch
import models.networks as nets
import models.losses as losses
import numpy as np
import torch.nn as nn

class ICSCN:
    def __init__(self, opt, device,is_train=False) -> None:
        self.generator = nets.Generator(opt.input_nc, opt.output_nc)
        self.generator.apply(self.weights_init_normal)
        self.generator.to(device)
        # Generator optimizer and losses
        self.is_train = is_train
        if self.is_train:
            self.generator_optim = torch.optim.Adam(self.generator.parameters(),
                                                    lr=opt.lr)
            self.adv_criterion = losses.AdvLoss().to(device)
            self.fm_criterion = losses.FMLoss().to(device)
            self.vgg_criterion = losses.VGGLoss().to(device)

        # Discriminator arch, optimizer and loss
        if self.is_train:
            self.discriminator = nets.Discriminator(opt.output_nc)
            self.discriminator.to(device)
            self.discriminator_optim = torch.optim.Adam(self.discriminator.parameters(),
                                                        lr=opt.lr)
            self.discriminator.apply(self.weights_init_normal)
            self.disc_criterion = losses.DiscLoss().to(device)

    def forward(self, clear_image: torch.Tensor,
                rainy_image: torch.Tensor) -> None:
        # De-rained image
        self.derained_image = self.generator(rainy_image)

        # Clear image
        self.clear_image = clear_image

    def forward_discriminator(self):
        if self.is_train:
            self.pred_rainy = self.discriminator(self.derained_image.detach())
            self.pred_clear = self.discriminator(self.clear_image)
    
    def optimize(self, opt) -> None:
        fake = torch.cat((self.derained_image, self.clear_image), 1)
        pred_rainy = self.discriminator(fake.detach())
        real = torch.cat((self.clear_image, self.clear_image), 1)
        pred_clear = self.discriminator(real)
        self.discriminator_optim.zero_grad()
        self.disc_loss = self.disc_criterion(pred_clear[-1],
                                             pred_rainy[-1])
        self.disc_loss.backward(retain_graph=True)

        # Discriminator update
        self.discriminator_optim.step()

        fake = torch.cat((self.clear_image, self.derained_image), 1)
        pred_rainy = self.discriminator(fake)
        pred_clear = self.discriminator(real)
        self.generator_optim.zero_grad()
        self.adv_loss = self.adv_criterion(pred_rainy[-1])
        self.vgg_loss = self.vgg_criterion(self.clear_image,
                                           self.derained_image)
        self.fm_loss = self.fm_criterion(pred_clear[:-1],
                                         pred_rainy[:-1])
        
        self.gen_loss = (opt.lambda_adv * self.adv_loss
                         + opt.lambda_vgg * self.vgg_loss
                         + opt.lambda_fm * self.fm_loss)
        self.gen_loss.backward()

        # Generator update
        self.generator_optim.step()

        
    def get_derained_image(self) -> torch.Tensor:
        return self.derained_image

    def weights_init_normal(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        


















