import torch
from torch import nn

'''
Inspired by https://github.com/aimerykong/OpenGAN/
'''


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)    

class Generator(nn.Module):
    '''
    Basic architecture for the generator. It is a ConvNet going from the latent dimension to the input space.
    The input to the generator is a vector in the specified latent dimension treated as a 3D tensor of 1x1 spatial dimension and latent_dim channels.

    Constructor parameters
    -------------
    latent_dim: the dimensione of the latent space whose vectors are input to this network
    base_width: the base width of each convolutional layer. Starting from the initial layer, the outputs have all a number of channels multiple of base_width.
    num_channels: number of channels of the output of the model
    '''
    def __init__(self, latent_dim:int=100, base_width:int=64, num_channels:int=512):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.base_width = base_width
        self.num_channels = num_channels
        
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.Conv2d(self.latent_dim, self.base_width * 8, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.base_width * 8),
            nn.ReLU(True),
            # state size. (self.base_width*8) x 4 x 4
            nn.Conv2d(self.base_width * 8, self.base_width * 4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.base_width * 4),
            nn.ReLU(True),
            # state size. (self.base_width*4) x 8 x 8
            nn.Conv2d( self.base_width * 4, self.base_width * 2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.base_width * 2),
            nn.ReLU(True),
            # state size. (self.base_width*2) x 16 x 16
            nn.Conv2d( self.base_width * 2, self.base_width*4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.base_width*4),
            nn.ReLU(True),
            # state size. (self.base_width) x 32 x 32
            nn.Conv2d( self.base_width*4, self.num_channels, 1, 1, 0, bias=True),
            #nn.Tanh()
            # state size. (self.num_channels) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class DiscriminatorFunnel(nn.Module):
    '''
    A binary classifier taking as input a datapoint. It classifies whether the datapoint is real or synthetic. It is a CNN.

    Constructor parameters
    -------------
    num_channels: number of channels of the input data
    base_width: the base width of each convolutional layer. Starting from the initial layer, the outputs have all a number of channels multiple of base_width.
    '''
    def __init__(self, num_channels=512, base_width=64):
        super(DiscriminatorFunnel, self).__init__()
        self.num_channels = num_channels
        self.base_width = base_width
        self.main = nn.Sequential(
            nn.Conv2d(self.num_channels, self.base_width*8, 3, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.base_width*8, self.base_width*4, 3, 1, 0, bias=False),
            nn.BatchNorm2d(self.base_width*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.base_width*4, self.base_width*2, 3, 1, 0, bias=False),
            nn.BatchNorm2d(self.base_width*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.base_width*2, self.base_width, 2, 1, 0, bias=False),
            nn.BatchNorm2d(self.base_width),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.base_width, 1, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)