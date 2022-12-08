import torch
import torch.nn as nn

# Adversarial loss
class AdvLoss(nn.Module):
    def __init__(self):
        super(AdvLoss, self).__init__()
        self.criterion = nn.MSELoss()
        self.register_buffer('label', torch.tensor(1.0))

    def forward(self, derained_img):
        return self.criterion(derained_img,
                              self.label.expand_as(derained_img))

# Discriminator loss
class DiscLoss(nn.Module):
    def __init__(self):
        super(DiscLoss, self).__init__()
        self.criterion = nn.MSELoss()
        self.register_buffer('clear_label', torch.tensor(1.0))
        self.register_buffer('derained_label', torch.tensor(0.0))

    def forward(self, clear_pred, derained_pred):
        clear_loss = self.criterion(clear_pred,
                                    self.clear_label.expand_as(clear_pred))
        derained_loss = self.criterion(derained_pred,
                                       self.derained_label.expand_as(derained_pred))

        return clear_loss + derained_loss
        
# Feature Matching loss
class FMLoss(nn.Module):
    def __init__(self):
        super(FMLoss, self).__init__()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/8.0, 1.0/4.0, 1.0/2.0, 1.0]

    def forward(self, clear_feat_list, derained_feat_list):
        loss = 0
        for idx in range(len(clear_feat_list)):
            loss += self.weights[idx] * self.criterion(clear_feat_list[idx],
                                                       derained_feat_list[idx])
        return loss

# VGG loss (perceptual loss)
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/16.0, 1.0/8.0, 1.0/4.0, 1.0/2.0, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss




from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(weights='DEFAULT').features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out



