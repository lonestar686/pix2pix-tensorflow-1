
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation, LeakyReLU
from tensorflow.keras.layers import Concatenate, Dropout

# generator uses unet architecture
class UnetGenerator(Model):
    """ to generate the data
    """
    def __init__(self, generator_outputs_channels, ngf):
        super(UnetGenerator, self).__init__()

        #
        self.ngf = ngf
        self.generator_outputs_channels = generator_outputs_channels
        #
        self.encoder = self._build_encoder(ngf)
        self.decoder = self._build_decoder(ngf, generator_outputs_channels)

    def call(self, x):

        # cache for convenience
        encoder = self.encoder
        decoder = self.decoder

        layers = []

        # encoder
        for op in encoder:
            x = op(x)
            layers.append(x)

        # decoder
        num_encoder_layers = len(layers)
        for decoder_layer, op in enumerate(decoder):
            skip_layer = num_encoder_layers - decoder_layer - 1
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                x = op(x)
            else:
                x = Concatenate(axis=-1)([x, layers[skip_layer]])
                x = op(x)

        return x

    def _build_encoder(self, ngf, a=0.2):

        encoder_network = []

        # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
        # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
        encoder_network += [
            Conv2D(ngf, kernel_size=(4, 4), strides=(2, 2), padding='same'),
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
                _UnetEncoderMid(out_channels)
             ]

        return encoder_network
    
    def _build_decoder(self, ngf, generator_outputs_channels):

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

        # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
        decoder_network += [
            _UnetDecoderEnd(generator_outputs_channels)
        ]

        return decoder_network

class _UnetEncoderMid(Model):
    def __init__(self, out_channels):
        super(_UnetEncoderMid, self).__init__()

        #
        self.leaky = LeakyReLU(alpha=0.2)
        # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
        self.conv = Conv2D(out_channels, kernel_size=(4, 4), strides=(2, 2), padding='same')
        self.bn = BatchNormalization()

    def call(self, x):
        x = self.leaky(x)
        x = self.conv(x)
        x = self.bn(x)
        return x

class _UnetDecoderMid(Model):
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

    def call(self, x):
        x = self.act(x)
        x = self.conv_trans(x)
        x = self.bn(x)
        #
        if self.dropout != None:
            x = self.dropout(x)

        return x

class _UnetDecoderEnd(Model):
    def __init__(self, out_channels):
        super(_UnetDecoderEnd, self).__init__()

        # 
        self.act1 = Activation('relu')
        # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
        self.conv_trans = Conv2DTranspose(out_channels, kernel_size=(4, 4), strides=(2, 2), padding='same')
        self.act2 =  Activation('tanh')

    def call(self, x):
        x = self.act1(x)
        x = self.conv_trans(x)
        x = self.act2(x)

        return x
    
# the discriminator
class UnetDiscriminator(Model):
    """ implement discriminator for shared nodes
    """
    def __init__(self, ndf, n_layers=3):
        super(UnetDiscriminator, self).__init__()
        #
        self.network = self._build_network(ndf, n_layers)

    def call(self, x):

        # apply operators to the data
        for op in self.network:
            x = op(x)

        return x

    def _build_network(self, ndf, n_layers):

        network = []

        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
        network += [
            Conv2D(ndf, kernel_size=(4, 4), strides=(2, 2), padding='same'),
            LeakyReLU(alpha=0.2),
        ]

        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        for i in range(n_layers):
            out_channels = ndf * min(2**(i+1), 8)
            stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            network += [
                Conv2D(out_channels, kernel_size=(4, 4), strides=(stride, stride), padding='same'),
                BatchNormalization(),
                LeakyReLU(alpha=0.2),             
            ]

        # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
        network += [
            Conv2D(1, kernel_size=(4, 4), strides=(1, 1), padding='same'),
            Activation('sigmoid')
        ]

        return network
