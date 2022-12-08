import argparse

class GeneralOptions:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--inputnc', dest='input_nc',
                                 type=int, default=3,
                                 help='number of input channels')
        self.parser.add_argument('--outputnc', dest='output_nc',
                                 type=int, default=3,
                                 help='number of output channels')
        self.parser.add_argument('--inputdir', dest='input_dir',
                                 type=str,
                                 help='path to the input directory')
        self.parser.add_argument('--outputdir', dest='output_dir',
                                 type=str,
                                 help='path to the output directory')

    def get_options(self):
        return self.parser.parse_args()


class TrainingOptions(GeneralOptions):
    def __init__(self) -> None:
        super(TrainingOptions, self).__init__()
        # Paths for the images used in the training regime
        self.parser.add_argument('--savedir', dest='save_dir',
                                 type=str,
                                 help='pat')
        self.parser.add_argument('--cleardir', dest='clear_dir',
                                 type=str,
                                 help='path to the clear images')
        self.parser.add_argument('--rainydir', dest='rainy_dir',
                                 type=str,
                                 help='path to the rainy images')
        # Training options
        self.parser.add_argument('--nepochs', dest='n_epochs',
                                 type=int, default=1000,
                                 help=''' number of epochs used
                                          in the training regime''')
        self.parser.add_argument('--lr',
                                 type=float, default=0.0002,
                                 help='''inital learning rate to be
                                 used in the training regime''')
        self.parser.add_argument('--lambdaadv', dest='lambda_adv',
                                 type=float, default=1.0,
                                 help='importance of the adversarial loss')
        self.parser.add_argument('--lambdavgg', dest='lambda_vgg',
                                 type=float, default=1.0,
                                 help='importance of the perceptual loss')
        self.parser.add_argument('--lambdafm', dest='lambda_fm',
                                 type=float, default=1.0,
                                 help='importance of the feature matching loss')
        self.parser.add_argument('--batchsize', dest='batch_size',
                                 type=int, default=12,
                                 help='''batch size used in the 
                                 training regime''')


