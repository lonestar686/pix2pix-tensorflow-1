import tensorflow as tf
import collections
from network_ops import *

EPS = 1e-12

__all__ = []

# decorator for import *
def export(obj):
    __all__.append(obj.__name__)
    return obj

#
Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")

@export
class pix2pix:
    """ to build pix2pix model
    """
    def __init__(self, a):
        self.a  = a 

    def create_model(self, inputs, targets):

        # create generator
        generator = Generator(self.a.ngf)

        #
        with tf.variable_scope("var_generator"):
            out_channels = int(targets.get_shape()[-1])
            outputs = generator.create_generator(inputs, out_channels)

        # create discriminators
        discriminator = Discriminator(self.a.ndf)

        # create two copies of discriminator, one for real pairs and one for fake pairs
        # they share the same underlying variables
        # with tf.name_scope("real_discriminator"):
        # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
        with tf.variable_scope("var_discriminator"):
            predict_real = discriminator.create_discriminator(inputs, targets)

        # with tf.name_scope("fake_discriminator"):
        # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
        with tf.variable_scope("var_discriminator"):
            predict_fake = discriminator.create_discriminator(inputs, outputs)   

        with tf.name_scope("discriminator_loss"):
            # minimizing -tf.log will try to get inputs to 1
            # predict_real => 1
            # predict_fake => 0
            discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

        with tf.name_scope("generator_loss"):
            # predict_fake => 1
            # abs(targets - outputs) => 0
            gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
            gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
            gen_loss = gen_loss_GAN * self.a.gan_weight + gen_loss_L1 * self.a.l1_weight

        with tf.name_scope("discriminator_train"):
            discrim_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='var_discriminator')
            print('--- discriminator ----')
            for i, var in enumerate(discrim_tvars):
                print('i={}, var = {}'.format(i, var))

            discrim_optim = tf.train.AdamOptimizer(self.a.lr, self.a.beta1)
            discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
            discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

        with tf.name_scope("generator_train"):
            with tf.control_dependencies([discrim_train]):
                gen_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='var_generator')
                print('--- generator ----')
                for i, var in enumerate(gen_tvars):
                    print('i={}, var = {}'.format(i, var))

                gen_optim = tf.train.AdamOptimizer(self.a.lr, self.a.beta1)
                gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
                gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

        ema = tf.train.ExponentialMovingAverage(decay=0.99)
        update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

        global_step = tf.train.get_or_create_global_step()
        incr_global_step = tf.assign(global_step, global_step+1)

        return Model(
            predict_real=predict_real,
            predict_fake=predict_fake,
            discrim_loss=ema.average(discrim_loss),
            discrim_grads_and_vars=discrim_grads_and_vars,
            gen_loss_GAN=ema.average(gen_loss_GAN),
            gen_loss_L1=ema.average(gen_loss_L1),
            gen_grads_and_vars=gen_grads_and_vars,
            outputs=outputs,
            train=tf.group(update_losses, incr_global_step, gen_train),
        )

class Generator:
    """ to generate the data
    """
    def __init__(self, ngf):
        self.ngf = ngf

    def create_generator(self, generator_inputs, generator_outputs_channels):

        ngf = self.ngf
        #
        encoder = self._build_encoder(ngf)
        decoder = self._build_decoder(ngf, generator_outputs_channels)

        # input data
        input_generator = tf.keras.layers.Input(tensor=generator_inputs)

        layers = []

        # encoder
        x = input_generator
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

        # # for qc testing
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
class Discriminator:
    """ implement discriminator for shared nodes
    """
    def __init__(self, ndf, n_layers=3):
        self.network = self._build_network(ndf, n_layers)

    def create_discriminator(self, discrim_inputs, discrim_targets):

        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        input_combined = tf.concat([discrim_inputs, discrim_targets], axis=-1)
        input_discriminator = tf.keras.layers.Input(tensor=input_combined)

        # apply operators to the data
        x = self.network(input_discriminator)

        # # for qc testing
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
