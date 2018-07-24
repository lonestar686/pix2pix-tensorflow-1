import tensorflow as tf
import collections

EPS = 1e-12

__all__ = []

# decorator for import *
def export(obj):
    __all__.append(obj.__name__)
    return obj

#
Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")

def lrelu(x, a=0.3):
    with tf.name_scope("lrelu"):
        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return tf.keras.layers.LeakyReLU(a)(x)

def batchnorm(inputs):
    return tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(inputs)

@export
class pix2pix:
    """ to build pix2pix model
    """
    def __init__(self, a):
        self.a  = a 

    def discrim_conv(self, batch_input, out_channels, stride):
        conv=tf.keras.layers.Conv2D(out_channels, kernel_size=4, strides=(stride, stride), padding='same', use_bias=True, kernel_initializer='glorot_uniform')
        return conv(batch_input)


    def gen_conv(self, batch_input, out_channels):
        # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
        if self.a.separable_conv:
            conv=tf.keras.layers.SeparableConv2D(out_channels, kernel_size=4, strides=(2,2), padding='same')
            return conv(batch_input)
        else:
            conv=tf.keras.layers.Conv2D(out_channels, kernel_size=4, strides=(2, 2), padding='same', use_bias=True, kernel_initializer='glorot_uniform')
            return conv(batch_input)


    def gen_deconv(self, batch_input, out_channels):
        # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
        if self.a.separable_conv:
            up = tf.keras.layers.UpSampling2D(size=(2, 2), data_format=None)(batch_input)
            conv=tf.keras.layers.SeparableConv2D(out_channels, kernel_size=4, strides=(1,1), padding='same')
            return conv(up)
        else:
            conv_trans = tf.keras.layers.Conv2DTranspose(out_channels, kernel_size=4, strides=(2, 2), padding='same', kernel_initializer='glorot_uniform')
            return conv_trans(batch_input)

    def create_generator(self, generator_inputs, generator_outputs_channels):
        layers = []

        # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
        with tf.variable_scope("encoder_1"):
            output = self.gen_conv(generator_inputs, self.a.ngf)
            layers.append(output)

        layer_specs = [
            self.a.ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
            self.a.ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
            self.a.ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
            self.a.ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
            self.a.ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
            self.a.ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
            self.a.ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
        ]

        for out_channels in layer_specs:
            with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
                rectified = lrelu(layers[-1], 0.2)
                # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                convolved = self.gen_conv(rectified, out_channels)
                output = batchnorm(convolved)
                layers.append(output)

        layer_specs = [
            (self.a.ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
            (self.a.ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
            (self.a.ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
            (self.a.ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
            (self.a.ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
            (self.a.ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
            (self.a.ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
        ]

        num_encoder_layers = len(layers)
        for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
            skip_layer = num_encoder_layers - decoder_layer - 1
            with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
                if decoder_layer == 0:
                    # first decoder layer doesn't have skip connections
                    # since it is directly connected to the skip_layer
                    input = layers[-1]
                else:
                    input = tf.keras.layers.Concatenate(axis=-1)([layers[-1], layers[skip_layer])

                rectified = tf.keras.activations.relu(input, alpha=0.0, max_value=None)
                # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
                output = self.gen_deconv(rectified, out_channels)
                output = batchnorm(output)

                if dropout > 0.0:
                    tf.keras.layers.Dropout(dropout, noise_shape=None, seed=None)(output)

                layers.append(output)

        # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
        with tf.variable_scope("decoder_1"):
            input = tf.keras.layers.Concatenate(axis=-1)([layers[-1], layers[0]])
            rectified = tf.keras.activations.relu(input, alpha=0.0, max_value=None)
            output = self.gen_deconv(rectified, generator_outputs_channels)
            output = tf.keras.activations.tanh(output)
            layers.append(output)

        return layers[-1]


    def create_model(self, inputs, targets):
        def create_discriminator(discrim_inputs, discrim_targets):
            n_layers = 3
            layers = []

            # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
            input = tf.concat([discrim_inputs, discrim_targets], axis=3)

            # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
            with tf.variable_scope("layer_1"):
                convolved = self.discrim_conv(input, self.a.ndf, stride=2)
                rectified = lrelu(convolved, 0.2)
                layers.append(rectified)

            # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
            # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
            # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
            for i in range(n_layers):
                with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                    out_channels = self.a.ndf * min(2**(i+1), 8)
                    stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                    convolved = self.discrim_conv(layers[-1], out_channels, stride=stride)
                    normalized = batchnorm(convolved)
                    rectified = lrelu(normalized, 0.2)
                    layers.append(rectified)

            # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                convolved = self.discrim_conv(rectified, out_channels=1, stride=1)
                output = tf.keras.activations.sigmoid(convolved)
                layers.append(output)

            return layers[-1]

        with tf.variable_scope("generator"):
            out_channels = int(targets.get_shape()[-1])
            outputs = self.create_generator(inputs, out_channels)

        # create two copies of discriminator, one for real pairs and one for fake pairs
        # they share the same underlying variables
        with tf.name_scope("real_discriminator"):
            with tf.variable_scope("discriminator"):
                # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
                predict_real = create_discriminator(inputs, targets)

        with tf.name_scope("fake_discriminator"):
            with tf.variable_scope("discriminator", reuse=True):
                # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
                predict_fake = create_discriminator(inputs, outputs)

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
            discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
            discrim_optim = tf.train.AdamOptimizer(self.a.lr, self.a.beta1)
            discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
            discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

        with tf.name_scope("generator_train"):
            with tf.control_dependencies([discrim_train]):
                gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
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
