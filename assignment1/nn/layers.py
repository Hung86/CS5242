import numpy as np
from nn.initializers import *
from nn.operators import *


class Layer(object):
    """
    Layer abstraction
    """

    def __init__(self, name):
        """Initialization"""
        self.name = name
        self.training = True  # The phrase, if for training then true
        self.trainable = False  # Whether there are parameters in this layer that can be trained

    def forward(self, input):
        """Forward pass, reture output"""
        raise NotImplementedError

    def backward(self, out_grad, input):
        """Backward pass, return gradient to input"""
        raise NotImplementedError

    def update(self, optimizer):
        """Update parameters in this layer"""
        pass

    def set_mode(self, training):
        """Set the phrase/mode into training (True) or tesing (False)"""
        self.training = training

    def set_trainable(self, trainable):
        """Set the layer can be trainable (True) or not (False)"""
        self.trainable = trainable

    def get_params(self, prefix):
        """Reture parameters and gradient of this layer"""
        return None


class Linear(Layer):
    def __init__(self, in_features, out_features, name='linear', initializer=Gaussian()):
        """Initialization

        # Arguments
            in_features: int, the number of input features
            out_features: int, the numbet of required output features
            initializer: Initializer class, to initialize weights
        """
        super(Linear, self).__init__(name=name)
        self.linear = linear()

        self.trainable = True

        self.weights = initializer.initialize((in_features, out_features))
        self.bias = np.zeros(out_features)

        self.w_grad = np.zeros(self.weights.shape)
        self.b_grad = np.zeros(self.bias.shape)

    def forward(self, input):
        output = self.linear.forward(input, self.weights, self.bias)
        return output

    def backward(self, out_grad, input):
        in_grad, self.w_grad, self.b_grad = self.linear.backward(
            out_grad, input, self.weights, self.bias)
        return in_grad

    def update(self, params):
        """Update parameters (self.weights and self.bias) with new params

        # Arguments
            params: dictionary, one key contains 'weights' and the other contains 'bias'

        # Returns
            none
        """
        for k, v in params.items():
            if 'weights' in k:
                self.weights = v
            else:
                self.bias = v

    def get_params(self, prefix):
        """Return parameters (self.weights and self.bias) as well as gradient (self.w_grad and self.b_grad)

        # Arguments
            prefix: string, to contruct prefix of keys in the dictionary (usually is the layer-ith)

        # Returns
            params: dictionary, store parameters of this layer, one key contains 'weights' and the other contains 'bias'
            grads: dictionary, store gradient of this layer, one key contains 'weights' and the other contains 'bias'

            None: if not trainable
        """
        if self.trainable:
            params = {
                prefix+':'+self.name+'/weights': self.weights,
                prefix+':'+self.name+'/bias': self.bias
            }
            grads = {
                prefix+':'+self.name+'/weights': self.w_grad,
                prefix+':'+self.name+'/bias': self.b_grad
            }
            return params, grads
        else:
            return None

class ReLU(Layer):
    def __init__(self, name='relu'):
        """Initialization
        """
        super(ReLU, self).__init__(name=name)
        self.relu = relu()

    def forward(self, input):
        """Forward pass

        # Arguments
            input: numpy array

        # Returns
            output: numpy array
        """
        output = self.relu.forward(input)
        return output

    def backward(self, out_grad, input):
        """Backward pass

        # Arguments
            out_grad: numpy array, gradient to output
            input: numpy array, same with forward input

        # Returns
            in_grad: numpy array, gradient to input 
        """
        in_grad = self.relu.backward(out_grad, input)
        return in_grad

class Leaky_ReLU(Layer):
    def __init__(self, alpha = 0.01, name='leaky_relu'):
        """Initialization
        """
        super(Leaky_ReLU, self).__init__(name=name)
        # alpha: Float >= 0. Negative slope coefficient. Default to 0.01.
        self.leaky_relu = leaky_relu(alpha)

    def forward(self, input):
        """Forward pass

        # Arguments
            input: numpy array

        # Returns
            output: numpy array
        """
        output = self.leaky_relu.forward(input)
        return output

    def backward(self, out_grad, input):
        """Backward pass

        # Arguments
            out_grad: numpy array, gradient to output
            input: numpy array, same with forward input

        # Returns
            in_grad: numpy array, gradient to input 
        """
        in_grad = self.leaky_relu.backward(out_grad, input)
        return in_grad

