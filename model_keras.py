import tensorflow as tf
import collections

# unet
from unet_model import UnetGenerator as Generator
from unet_model import UnetDiscriminator as Discriminator

# Resnet
# from resnet_model import ResnetGenerator as Generator
# from resnet_model import NLayerDiscriminator as Discriminator

EPS = 1e-12

__all__ = []

# decorator for import *
def export(obj):
    __all__.append(obj.__name__)
    return obj

#
Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, \
                                         discrim_loss, discrim_grads_and_vars, \
                                         gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, \
                                         train")

@export
class pix2pix:
    """ to build pix2pix model
    """
    def __init__(self, a):
        self.a  = a

        # create a discriminator for sharing
        self.discriminator = Discriminator(self.a.ndf)

    def create_generator(self, inputs, outputs_channels):
        """ function to create a generator
        """
        # input data
        inputs_generator = tf.keras.layers.Input(tensor=inputs)

        # create generator
        generator = Generator(outputs_channels, self.a.ngf)

        # apply the network
        with tf.variable_scope("var_generator"):
            outputs = generator(inputs_generator)

        return outputs

    def create_discriminator(self, inputs, targets):
        """ function to create a discriminator
        """
        # create two copies of discriminator, one for real pairs and one for fake pairs
        # they share the same underlying variables
        # with tf.name_scope("real_discriminator"):
        # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
        with tf.variable_scope("var_discriminator"):
            # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
            inputs_targets = tf.concat([inputs, targets], axis=-1)
            #
            inputs_discriminator = tf.keras.layers.Input(tensor=inputs_targets)
            # apply the network
            predict = self.discriminator(inputs_discriminator)

        return predict

    def create_model(self, inputs, targets):

        # create generator
        out_channels = int(targets.get_shape()[-1])
        outputs = self.create_generator(inputs, out_channels)

        # create two copies of discriminator, one for real pairs and one for fake pairs
        # they share the same underlying variables
        # real_discriminator
        predict_real = self.create_discriminator(inputs, targets)

        # fake_discriminator           
        predict_fake = self.create_discriminator(inputs, outputs)   

        # 
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

        # for keras batch normalization?
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.name_scope("discriminator_train"):
            with tf.control_dependencies(extra_update_ops):
                discrim_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='var_discriminator')
                discrim_optim = tf.train.AdamOptimizer(self.a.lr, self.a.beta1)
                discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
                discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

        with tf.name_scope("generator_train"):
            with tf.control_dependencies([discrim_train]):
                gen_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='var_generator')
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
