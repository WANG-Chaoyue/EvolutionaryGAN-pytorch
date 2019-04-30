import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

class DCGANGenerator_cifar10(nn.Module):
    def __init__(self, z_dim, ngf=64, output_nc=3,  norm_layer=nn.BatchNorm2d):
        super(DCGANGenerator_cifar10, self).__init__()
        self.z_dim = z_dim
        self.ngf = ngf
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        seq = [nn.ConvTranspose2d(z_dim, ngf*8, 4, stride=1, padding=0, bias=use_bias),
               norm_layer(ngf*8),
               nn.ReLU(),
               nn.ConvTranspose2d(ngf*8, ngf*4, 4, stride=2, padding=(1,1), bias=use_bias),
               norm_layer(ngf*4),
               nn.ReLU(),
               nn.ConvTranspose2d(ngf*4, ngf*2, 4, stride=2, padding=(1,1), bias=use_bias),
               norm_layer(ngf*2),
               nn.ReLU(),
               nn.ConvTranspose2d(ngf*2, ngf, 4, stride=2, padding=(1,1), bias=use_bias),
               norm_layer(ngf),
               nn.ReLU(),
               nn.ConvTranspose2d(ngf, output_nc, 3, stride=1, padding=(1,1)),
               nn.Tanh()]

        self.model = nn.Sequential(*seq)

    def forward(self, input):
        return self.model(input.view(-1, self.z_dim, 1, 1))


class DCGANDiscriminator_cifar10(nn.Module):
    def __init__(self, ndf=64, input_nc=3, norm_layer=nn.BatchNorm2d):
        super(DCGANDiscriminator_cifar10, self).__init__()

        self.ndf = ndf
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        seq = [nn.Conv2d(input_nc, ndf, 3, stride=1, padding=(1,1), bias=use_bias),
               nn.LeakyReLU(0.2),
               nn.Conv2d(ndf, ndf*2, 4, stride=2, padding=(1,1), bias=use_bias),
               norm_layer(ndf*2),
               nn.LeakyReLU(0.2),
               nn.Conv2d(ndf*2, ndf*4, 4, stride=2, padding=(1,1), bias=use_bias),
               norm_layer(ndf*4),
               nn.LeakyReLU(0.2),
               nn.Conv2d(ndf*4, ndf*8, 4, stride=2, padding=(1,1), bias=use_bias),
               norm_layer(ndf*8),
               nn.LeakyReLU(0.2)]
        
        self.cnn_model = nn.Sequential(*seq)

        fc = [nn.Linear(4*4*ndf*8, 1)]
        self.fc = nn.Sequential(*fc)

    def forward(self, input):
        x = self.cnn_model(input)
        x = x.view(-1, 4*4*self.ndf*8)
        x = self.fc(x)
        return(x)

