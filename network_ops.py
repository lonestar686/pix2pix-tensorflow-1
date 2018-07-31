# 
from network_wrapper_keras import *
from network_modules import Module, Sequential

# wrap common network operations into module wrappers
class Conv2d(Module):
    """ wrapper for convolution
    """
    def __init__(self, out_channels, kernel_size, stride):

        self.conv=conv(out_channels, kernel_size, stride)

    def forward(self, x):
        x = self.conv(x)
        return x

class ConvTranspose2d(Module):
    """ wrapper for transposed convolution
    """
    def __init__(self, out_channels, kernel_size, stride):
        self.deconv = deconv(out_channels, kernel_size, stride)

    def forward(self, x):
        x = self.deconv(x)
        return x

class BatchNorm2d(Module):
    """ wrapper for batch normalization
    """
    def __init__(self):
        self.batchnorm = batchnorm()

    def forward(self, x):
        x = self.batchnorm(x)
        return x

class Activation(Module):
    """ wrapper for some activations
    """
    def __init__(self, act):
        self.activation = activation(act)
    
    def forward(self, x):
        x = self.activation(x)
        return x

class LeakyReLU(Module):
    """ wrapper for leaky ReLU
    """
    def __init__(self, a):
        self.lrelu = lrelu(a)
    
    def forward(self, x):
        x = self.lrelu(x)
        return x

class Dropout(Module):
    """ wrapper for dropout
    """
    def __init__(self, dropout_rate):
        self.dropout = dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout(x)
        return x

class Concat(Module):
    """ wrapper for concatenate
    """
    def __init__(self, axis):
        self.concat = concat(axis)

    def forward(self, x1, x2):
        x = self.concat([x1, x2])
        return x

class Identity(Module):
    """ wrapper for identity 
    """
    def forward(self, x):
        return x
