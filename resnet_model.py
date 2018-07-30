
from network_modules import Module, Sequential
from network_ops import Conv2d, ConvTranspose2d, BatchNorm2d
from network_ops import Activation, LeakyReLU, Dropout
import numpy as np

import tensorflow as tf

# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
class ResnetGenerator(Module):
    def __init__(self, output_nc, ngf=64, use_dropout=False, n_blocks=6):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.output_nc = output_nc
        self.ngf = ngf

        model = [Conv2d(ngf, kernel_size=7, stride=1),
                 BatchNorm2d(),
                 Activation('relu')]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [Conv2d(ngf * mult * 2, kernel_size=3,
                                stride=2),
                      BatchNorm2d(),
                      Activation('relu')]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, use_dropout=use_dropout)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [ConvTranspose2d(int(ngf * mult / 2),
                                         kernel_size=3, stride=2),
                      BatchNorm2d(),
                      Activation('relu')]

        model += [Conv2d(output_nc, kernel_size=7, stride=1)]
        model += [Activation('tanh')]

        self.model = Sequential(*model)

    def forward(self, input):

        # input data
        input = tf.keras.layers.Input(tensor=input)

        output = self.model(input)

        # # qc 
        # test_model = tf.keras.models.Model(input, output)
        # test_model.summary()

        return output

# Define a resnet block
class ResnetBlock(Module):
    def __init__(self, dim, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, use_dropout)

    def build_conv_block(self, dim, use_dropout):
        conv_block = []

        conv_block += [Conv2d(dim, kernel_size=3, stride=1),
                       BatchNorm2d(),
                       Activation('relu')]
        if use_dropout:
            conv_block += [Dropout(0.5)]

        conv_block += [Conv2d(dim, kernel_size=3, stride=1),
                       BatchNorm2d()]

        return Sequential(*conv_block)

    def forward(self, x):
        out = tf.keras.layers.Add()([x, self.conv_block(x)])
        return out


# Defines the PatchGAN discriminator.
class NLayerDiscriminator(Module):
    def __init__(self, ndf=64, n_layers=3, 
                       use_sigmoid=True):
        super(NLayerDiscriminator, self).__init__()

        kw = 4
        sequence = [
            Conv2d(ndf, kernel_size=kw, stride=2),
            LeakyReLU(0.2)
        ]

        for n in range(1, n_layers):
            nf_mult = min(2**n, 8)
            sequence += [
                Conv2d(ndf * nf_mult, kernel_size=kw, stride=2), 
                BatchNorm2d(), 
                LeakyReLU(0.2)
            ]

        nf_mult = min(2**n_layers, 8)
        sequence += [
            Conv2d(ndf * nf_mult, kernel_size=kw, stride=1), 
            BatchNorm2d(), 
            LeakyReLU(0.2)
        ]

        sequence += [Conv2d(1, kernel_size=kw, stride=1)]

        if use_sigmoid:
            sequence += [Activation('sigmoid')]

        self.model = Sequential(*sequence)

    def forward(self, input):
        # input data
        input = tf.keras.layers.Input(tensor=input)
        output = self.model(input)

        # # qc 
        # test_model = tf.keras.models.Model(input, output)
        # test_model.summary()

        return output