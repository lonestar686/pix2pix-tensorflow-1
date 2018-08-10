
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation, LeakyReLU
from tensorflow.keras.layers import Concatenate, Dropout

from network_modules import Module

# generator uses unet architecture
class UnetGenerator(Module):
    """ to generate the data
    """
    def __init__(self, outputs_channels, ngf):
        super(UnetGenerator, self).__init__()
        #
        self.encoder = self._build_encoder(ngf)
        self.decoder = self._build_decoder(outputs_channels, ngf)

    def forward(self, x, is_training):

        # cache for convenience
        encoder = self.encoder
        decoder = self.decoder

        layers = []

        # encoder
        for op in encoder:
            x = op(x, is_training)
            layers.append(x)

        # decoder
        num_encoder_layers = len(layers)
        for decoder_layer, op in enumerate(decoder):
            skip_layer = num_encoder_layers - decoder_layer - 1
            # first decoder layer doesn't have skip connections
            # since it is directly connected to the skip_layer
            if decoder_layer == 0:
                x = op(x, is_training)
            else:
                x = Concatenate(axis=-1)([x, layers[skip_layer]])
                x = op(x, is_training)

        return x

    def _build_encoder(self, ngf, alpha=0.2):

        encoder_network = []

        # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
        # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
        encoder_network += [
            _UnetEncoderBeg(ngf)
        ]

        # other encorders
        layer_specs = [
            ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
            ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
            ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
            ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
            ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
            ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
            ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
        ]

        for out_channels in layer_specs:
            encoder_network += [ 
                _UnetEncoderMid(out_channels, alpha)
             ]

        return encoder_network
    
    def _build_decoder(self, outputs_channels, ngf):

        decoder_network = []

        # decoders
        layer_specs = [
            (ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
            (ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
            (ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
            (ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
            (ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
            (ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
            (ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
        ]

        # decoders
        for (out_channels, dropout) in layer_specs:
            # collecting operators
            decoder_network += [
                _UnetDecoderMid(out_channels, dropout)
            ]

        # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, outputs_channels]
        decoder_network += [
            _UnetDecoderEnd(outputs_channels)
        ]

        return decoder_network

class _UnetEncoderBeg(Module):
    def __init__(self, ngf):
        super(_UnetEncoderBeg, self).__init__()
        #
        self.conv = Conv2D(ngf, kernel_size=(4, 4), strides=(2, 2), padding='same')

    def forward(self, x, is_training):
        x = self.conv(x)
        return x

class _UnetEncoderMid(Module):
    def __init__(self, out_channels, alpha):
        super(_UnetEncoderMid, self).__init__()
        #
        self.leaky = LeakyReLU(alpha)
        # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
        self.conv = Conv2D(out_channels, kernel_size=(4, 4), strides=(2, 2), padding='same')
        self.bn = BatchNormalization()

    def forward(self, x, is_training):
        x = self.leaky(x)
        x = self.conv(x)
        x = self.bn(x, training=is_training)
        return x

class _UnetDecoderMid(Module):
    def __init__(self, out_channels, dropout):
        super(_UnetDecoderMid, self).__init__()
        # 
        self.act = Activation('relu')
        # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
        self.conv_trans = Conv2DTranspose(out_channels, kernel_size=(4, 4), strides=(2, 2), padding='same')
        self.bn = BatchNormalization()
        #
        self.dropout = None
        if dropout > 0.0:
            self.dropout = Dropout(dropout)

    def forward(self, x, is_training):
        x = self.act(x)
        x = self.conv_trans(x)
        x = self.bn(x, training=is_training)
        #
        if self.dropout is not None:
            x = self.dropout(x, training=is_training)

        return x

class _UnetDecoderEnd(Module):
    def __init__(self, out_channels):
        super(_UnetDecoderEnd, self).__init__()

        # 
        self.act = Activation('relu')
        # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
        self.conv_trans = Conv2DTranspose(out_channels, kernel_size=(4, 4), strides=(2, 2), padding='same')
        self.act =  Activation('tanh')

    def forward(self, x, is_training):   # Note: the flag 'is_training' is only for convenience
        x = self.act(x)
        x = self.conv_trans(x)
        x = self.act(x)

        return x
    
# the discriminator
class Discriminator(Module):
    """ implement discriminator for shared nodes
    """
    def __init__(self, ndf, n_layers=3):
        super(Discriminator, self).__init__()
        #
        self.network = self._build_network(ndf, n_layers)

    def forward(self, x, is_training):

        # apply operators to the data
        for op in self.network:
            x = op(x, is_training)

        return x

    def _build_network(self, ndf, n_layers):

        network = []

        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
        network += [
            _DiscriminatorBeg(ndf)
        ]

        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        for i in range(n_layers):
            out_channels = ndf * min(2**(i+1), 8)
            stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            network += [
                _DiscriminatorMid(out_channels, stride)     
            ]

        # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
        network += [
            _DiscriminatorEnd(1)
        ]

        return network

class _DiscriminatorBeg(Module):
    def __init__(self, out_channels, alpha=0.2):
        super(_DiscriminatorBeg, self).__init__()
        #
        self.conv = Conv2D(out_channels, kernel_size=(4, 4), strides=(2, 2), padding='same')
        self.leaky = LeakyReLU(alpha)     

    def forward(self, x, is_training):
        #
        x = self.conv(x)
        x = self.leaky(x)

        return x

class _DiscriminatorMid(Module):
    def __init__(self, out_channels, stride, alpha=0.2):
        super(_DiscriminatorMid, self).__init__()
        #
        self.conv = Conv2D(out_channels, kernel_size=(4, 4), strides=(stride, stride), padding='same')
        self.bn = BatchNormalization()
        self.leaky = LeakyReLU(alpha)     

    def forward(self, x, is_training):
        #
        x = self.conv(x)
        x = self.bn(x, training=is_training)
        x = self.leaky(x)

        return x

class _DiscriminatorEnd(Module):
    def __init__(self, out_channels):
        super(_DiscriminatorEnd, self).__init__()

        self.conv = Conv2D(out_channels, kernel_size=(4, 4), strides=(1, 1), padding='same')
        self.act = Activation('sigmoid')

    def forward(self, x, is_training):   # Note: the flag 'is_training' is only for convenience
        x = self.conv(x)
        x = self.act(x)

        return x
