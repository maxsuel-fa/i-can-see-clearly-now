import torch
import torch.nn as nn

# General convolutional layer
class ConvLayer(nn.Module):
    def __init__(self, input_nc, output_nc, conv_type='down_conv', conv_kernel_size=3, conv_stride=2,
                 conv_padding=1, conv_output_padding=1, norm_type=nn.InstanceNorm2d):
        super(ConvLayer, self).__init__()

        layer = []
        if conv_type == 'up_conv':
            layer += [nn.ConvTranspose2d(input_nc, output_nc, conv_kernel_size, stride=conv_stride,
                                         padding=conv_padding, output_padding=conv_output_padding)]
        elif conv_type == 'down_conv':
            layer += [nn.Conv2d(input_nc, output_nc, conv_kernel_size,
                                stride=conv_stride, padding=conv_padding)]
        else:
            raise NotImplementedError('convolution [%s] is not implemented' % conv_type)

        layer += [norm_type(output_nc), nn.LeakyReLU(negative_slope=0.2, inplace=True)]
        self.layer = nn.Sequential(*layer)

    def forward(self, layer_input):
        return self.layer(layer_input)



# Residual network block
class ResNetBlock(nn.Module):
    def __init__(self, num_of_channels):
        super(ResNetBlock, self).__init__()

        self.conv_block = ConvLayer(num_of_channels, num_of_channels, 
                                conv_stride=1)

        # In adition to the convolution, norm block has also normaliztion
        norm_block = [nn.Dropout(0.5), nn.Conv2d(num_of_channels, num_of_channels, 
                                             kernel_size=3, padding=1),
                  nn.InstanceNorm2d(num_of_channels)]
        
        self.norm_block = nn.Sequential(*norm_block)

    def forward(self, res_block_input):
        conv_block_output = self.conv_block.forward(res_block_input)
        output = res_block_input + self.norm_block(conv_block_output)
        return output

# Generator architecture
class Generator(nn.Module):
    # Class attributes
    n_down_conv = 4
    n_up_conv = 4
    n_resnet_block = 9

    def __init__(self, input_nc, output_nc, ngf=32):
        super(Generator, self).__init__()

        # Number of input and output channels for intermediate layers
        input_channels = None
        output_channels = None

        # Down-convolutinal layers
        down_conv_layers = [ConvLayer(input_nc, ngf)]

        for i in range(self.n_down_conv - 1):
            input_channels = 2**i * ngf
            output_channels = 2 * input_channels
            down_conv_layers += [ConvLayer(input_channels, output_channels)]
        self.down_conv_layers = nn.Sequential(*down_conv_layers)

        # Residual network blocks
        resnet_blocks = []

        for i in range(self.n_resnet_block):
            resnet_blocks += [ResNetBlock(output_channels)]
        self.resnet_blocks = nn.Sequential(*resnet_blocks)

        # Up-convolutional layers
        up_conv_layers = []

        for i in range(self.n_up_conv - 1, 0, -1):
            input_channels = 2**i * ngf
            output_channels = int(input_channels / 2)
            if i == 1 or i == 2:
                input_channels *= 2
            up_conv_layers += [ConvLayer(input_channels,
                                              output_channels, conv_type='up_conv')]
        input_channels = output_channels
        up_conv_layers += [ConvLayer(input_channels, output_nc,
                                          conv_type='up_conv')]

        self.up_conv_layers = nn.Sequential(*up_conv_layers)

    def forward(self, gen_input):
        down_conv1 = self.down_conv_layers[0].forward(gen_input)
        down_conv2 = self.down_conv_layers[1].forward(down_conv1)
        down_conv3 = self.down_conv_layers[2].forward(down_conv2)
        down_conv4 = self.down_conv_layers[3].forward(down_conv3)
        
        res_output = down_conv4
        for i in range(self.n_resnet_block):
            res_output = self.resnet_blocks[i].forward(res_output)
        
        up_conv1 = self.up_conv_layers[0].forward(res_output)
        up_conv2 = self.up_conv_layers[1].forward(torch.cat((up_conv1, down_conv3), 1))
        up_conv3 = self.up_conv_layers[2].forward(torch.cat((up_conv2, down_conv2), 1))
        gen_output = self.up_conv_layers[3].forward(up_conv3)

        return gen_output

# The discriminator architecture
class Discriminator(nn.Module):
    # Including input and output layers
    n_layers = 5

    def __init__(self, input_nc, ndf=32):
        super(Discriminator, self).__init__()

        input_channels = None
        output_channels = None
        model = []

        # Adds not only the conv layers but also the 
        #input and output layers to the architecture
        for i in range(self.n_layers):
            if not i:
                model += [ConvLayer(input_nc, ndf, 
                                    conv_kernel_size=4, 
                                    conv_padding=1)]
            else:
                input_channels = 2**(i - 1) * ndf
                output_channels = (2 * input_channels 
                                   if i < self.n_layers - 1 else 1)
                stride = 2 if i < self.n_layers - 2 else 1
                model += [ConvLayer(input_channels, output_channels,
                                    conv_kernel_size=4, conv_stride=stride,
                                    conv_padding=1, norm_type=nn.BatchNorm2d)]
        self.model = nn.Sequential(*model)

    def forward(self, disc_input):
        output = disc_input
        output_list = []
        for layer in self.model:
            output = layer(output)
            output_list += [output]
        return output_list

