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

    def create_model(self, inputs, targets):

        # create generator
        out_channels = int(targets.get_shape()[-1])
        generator = Generator(out_channels, self.a.ngf)

        # apply the network
        with tf.variable_scope("var_generator"):
            outputs = generator(inputs)

        # create discriminators
        discriminator = Discriminator(self.a.ndf)

        # create two copies of discriminator, one for real pairs and one for fake pairs
        # they share the same underlying variables
        # with tf.name_scope("real_discriminator"):
        # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
        with tf.variable_scope("var_discriminator"):
            # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
            input_targets = tf.concat([inputs, targets], axis=-1)
            # apply the network
            predict_real = discriminator(input_targets)

        # with tf.name_scope("fake_discriminator"):
        # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
        with tf.variable_scope("var_discriminator"):
            # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
            input_outputs = tf.concat([inputs, outputs], axis=-1)
            # apply the network           
            predict_fake = discriminator(input_outputs)   

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
