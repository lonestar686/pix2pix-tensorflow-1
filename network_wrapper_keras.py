# utility functions for building unet
import tensorflow as tf

def identity():
    return tf.keras.layers.Lambda(lambda x: x)

# utility functions
def lrelu(a=0.2):
    return tf.keras.layers.LeakyReLU(a)

def activation(act):
    return tf.keras.layers.Activation(act)

def dropout(a):
    return tf.keras.layers.Dropout(a)

def concat(axis=-1):
    return tf.keras.layers.Concatenate(axis)

def batchnorm(axis=-1, momentum=0.99, epsilon=0.001):
    
    return tf.keras.layers.BatchNormalization(axis, momentum=momentum, epsilon=epsilon, 
                             gamma_initializer=tf.random_normal_initializer(1.0, 0.02))

def conv(out_channels, kernel_size, stride):
    initializer = tf.random_normal_initializer(0, 0.02)
    conv=tf.keras.layers.Conv2D(out_channels, 
                                kernel_size=kernel_size, 
                                strides=(stride, stride), 
                                padding='same', 
                                kernel_initializer=initializer)
    return conv

def deconv(out_channels, kernel_size, stride):
    initializer = tf.random_normal_initializer(0, 0.02)
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    conv_trans = tf.keras.layers.Conv2DTranspose(out_channels, 
                                                 kernel_size=kernel_size, 
                                                 strides=(stride, stride), 
                                                 padding='same', 
                                                 kernel_initializer=initializer)
    return conv_trans

#     # separable_conv:
# def gen_conv_sep(batch_input, out_channels):
#     # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
#     conv=tf.keras.layers.SeparableConv2D(out_channels, 
#                                          kernel_size=4, 
#                                          strides=(2,2), 
#                                          padding='same')
#     return conv(batch_input)

#     # separable_conv:
# def gen_deconv_sep(batch_input, out_channels):
#     # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
#     up = tf.keras.layers.UpSampling2D(size=(2, 2), data_format=None)(batch_input)
#     conv=tf.keras.layers.SeparableConv2D(out_channels, 
#                          kernel_size=4, 
#                          strides=(1,1), 
#                          padding='same')
#     return conv(up)