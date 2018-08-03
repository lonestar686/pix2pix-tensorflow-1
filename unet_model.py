
from network_ops import *

# generator uses unet architecture
class UnetGenerator(Module):
    """ to generate the data
    """
    def __init__(self, generator_outputs_channels, ngf):
        self.ngf = ngf
        self.generator_outputs_channels = generator_outputs_channels
        #
        self.encoder = self._build_encoder(ngf)
        self.decoder = self._build_decoder(ngf, generator_outputs_channels)

    def forward(self, inputs_generator):

        # cache for convenience
        encoder = self.encoder
        decoder = self.decoder

        layers = []

        # encoder
        x = inputs_generator
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
                x = Concat(axis=-1)(x, layers[skip_layer])
                x = op(x)

        # for qc testing
        # model = tf.keras.models.Model(input_generator, x)
        # model.summary()

        return x

    def _build_encoder(self, ngf, a=0.2):

        encoder_network = []

        # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
        # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
        encoder_network += [
            Conv2d(ngf, kernel_size=4, stride=2),
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
            encoder_network += [ Sequential(
                LeakyReLU(a=0.2),
                # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                Conv2d(out_channels, kernel_size=4, stride=2),
                BatchNorm2d())
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
            tmp = [
                Activation('relu'),
                # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
                ConvTranspose2d(out_channels, kernel_size=4, stride=2),
                BatchNorm2d()
            ]

            if dropout > 0.0:
                tmp += [Dropout(dropout)]

            decoder_network += [Sequential(*tmp)]    # Note: it's a list

        # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
        decoder_network += [ Sequential(
            Activation('relu'),
            ConvTranspose2d(generator_outputs_channels, kernel_size=4, stride=2),
            Activation('tanh'))
        ]

        return decoder_network

# the discriminator
class UnetDiscriminator(Module):
    """ implement discriminator for shared nodes
    """
    def __init__(self, ndf, n_layers=3):
        self.network = self._build_network(ndf, n_layers)

    def forward(self, inputs_discriminator):

        # apply operators to the data
        x = self.network(inputs_discriminator)

        # for qc testing
        # model = tf.keras.models.Model(input_discriminator, x)
        # model.summary()

        return x

    def _build_network(self, ndf, n_layers):

        network = []

        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
        network += [
            Conv2d(ndf, kernel_size=4, stride=2),
            LeakyReLU(a=0.2),
        ]

        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        for i in range(n_layers):
            out_channels = ndf * min(2**(i+1), 8)
            stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            network += [
                Conv2d(out_channels, kernel_size=4, stride=stride),
                BatchNorm2d(),
                LeakyReLU(a=0.2),             
            ]

        # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
        network += [
            Conv2d(1, kernel_size=4, stride=1), # TODO: need to recheck it
            Activation('sigmoid')
        ]

        return Sequential(*network)
