import torch
import torch.nn as nn

# General convolutional layer
class ConvLayer(nn.Module):
    def __init__(self, input_nc, output_nc, 
                 conv_type='down_conv', conv_kernel_size=3, conv_stride=2,
                 conv_padding=1, conv_output_padding=1, 
                 norm_type=nn.InstanceNorm2d):
        super(ConvLayer, self).__init__()

        layer = []
        if conv_type == 'up_conv':
            layer += [nn.ConvTranspose2d(input_nc, output_nc, 
                                         conv_kernel_size, stride=conv_stride,
                                         padding=conv_padding, 
                                         output_padding=conv_output_padding)]
        elif conv_type == 'down_conv':
            layer += [nn.Conv2d(input_nc, output_nc, conv_kernel_size,
                                stride=conv_stride, padding=conv_padding)]
        else:
            raise NotImplementedError('convolution [%s] is not implemented' % conv_type)

        layer += [norm_type(output_nc), nn.LeakyReLU(negative_slope=0.2, 
                                                     inplace=True)]
        self.layer = nn.Sequential(*layer)

    def forward(self, x):
        return self.layer(x)



# Residual network block
class ResNetBlock(nn.Module):
    def __init__(self, num_of_channels):
        super(ResNetBlock, self).__init__()

        self.conv_block = ConvLayer(num_of_channels, num_of_channels, 
                                conv_stride=1)

        norm_block = [nn.Conv2d(num_of_channels, num_of_channels, 
                                kernel_size=3, padding=1),
                      nn.InstanceNorm2d(num_of_channels)]
        
        self.norm_block = nn.Sequential(*norm_block)

    def forward(self, x):
        y = self.conv_block(x)
        y = x + self.norm_block(y)
        return y

# Generator architecture
class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, 
                 n_down_conv=4, n_resnet_block=9):
        super(Generator, self).__init__()

        # Number of input and output channels for intermediate layers
        input_channels = None
        output_channels = None

        # Down-convolutinal layers
        down_conv_layers = [ConvLayer(input_nc, ngf)]

        for i in range(n_down_conv - 1):
            input_channels = 2**i * ngf
            output_channels = 2 * input_channels
            down_conv_layers += [ConvLayer(input_channels, output_channels)]
        self.down_conv_layers = nn.Sequential(*down_conv_layers)

        # Residual network blocks
        resnet_blocks = []

        for i in range(n_resnet_block):
            resnet_blocks += [ResNetBlock(output_channels)]
        self.resnet_blocks = nn.Sequential(*resnet_blocks)

        # Up-convolutional layers
        up_conv_layers = []

        for i in range(n_down_conv - 1, 0, -1):
            input_channels = 2**i * ngf
            output_channels = int(input_channels / 2)
            if i == 1 or i == 2:
                input_channels *= 2
            up_conv_layers += [ConvLayer(input_channels,
                                         output_channels, 
                                         conv_type='up_conv')]
        input_channels = output_channels
        up_conv_layers += [ConvLayer(input_channels, output_nc,
                                          conv_type='up_conv')]

        self.up_conv_layers = nn.Sequential(*up_conv_layers)

    def forward(self, x):
        down_conv1 = self.down_conv_layers[0].forward(x)
        down_conv2 = self.down_conv_layers[1].forward(down_conv1)
        down_conv3 = self.down_conv_layers[2].forward(down_conv2)
        down_conv4 = self.down_conv_layers[3].forward(down_conv3)
        
        res_output = self.resnet_blocks(down_conv4)
        
        up_conv1 = self.up_conv_layers[0].forward(res_output)
        
        # Concatenates the output of the third down-convolutional layer
        # with the output of the first up-convolutional layer
        x = torch.cat((up_conv1, down_conv3), 1)
        up_conv2 = self.up_conv_layers[1].forward(x)
        
        # Concatenates the output of the second down-convolutional layer
        # with the output of the second up-convolutional layer
        x = torch.cat((up_conv2, down_conv2), 1)
        up_conv3 = self.up_conv_layers[2].forward(x)
        y = self.up_conv_layers[3].forward(up_conv3)

        return y

# The discriminator architecture
class Discriminator(nn.Module):
    def __init__(self, input_nc, output_nc, ndf=32, n_layers=5):
        super(Discriminator, self).__init__()
        model = [ConvLayer(input_nc + output_nc, ndf)]

        input_channels = None
        output_channels = None
        for i in range(1, n_layers - 1):
            input_channels = 2**(i - 1) * ndf
            output_channels = 2 * input_channels 
            stride = 2 if i < self.n_layers - 2 else 1

            model += [ConvLayer(input_channels, output_channels, 
                                conv_stride=stride)]
        model += [ConvLayer(input_channels, 1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        output = x
        output_list = []
        for layer in self.model:
            output = layer(output)
            output_list += [output]
        return output_list


