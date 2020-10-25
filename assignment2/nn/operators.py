import numpy as np
from utils.tools import *
from nn.functional import sigmoid
from itertools import product
# Attension:
# - Never change the value of input, which will change the result of backward

def img2col(data, h_indices, w_indices, k_h, k_w):
    """
    Convert convolution operation into a matrix product
    """
    batch = data.shape[0]
    #################### To do ####################
    _, c, w, h = data.shape
    h_len = len(h_indices)
    w_len = len(w_indices)
    
    def func(x):
        col = 0
        image2col_X = np.zeros((c*k_h*k_w, w_len*h_len))
        for h_i in h_indices:
            for w_i in w_indices:
                image2col_X[:, col] = x[0:c, h_i:h_i + k_h, w_i:w_i + k_w].flatten()
                col += 1
                
        return image2col_X
    
    out = np.stack(map(func, data), axis=0)
    ###############################################
    return out

class operator(object):
    """
    operator abstraction
    """

    def forward(self, input):
        """Forward operation, reture output"""
        raise NotImplementedError

    def backward(self, out_grad, input):
        """Backward operation, return gradient to input"""
        raise NotImplementedError


class relu(operator):
    def __init__(self):
        super(relu, self).__init__()

    def forward(self, input):
        output = np.maximum(0, input)
        return output

    def backward(self, out_grad, input):
        in_grad = (input >= 0) * out_grad
        return in_grad


class flatten(operator):
    def __init__(self):
        super(flatten, self).__init__()

    def forward(self, input):
        batch = input.shape[0]
        output = input.copy().reshape(batch, -1)
        return output

    def backward(self, out_grad, input):
        in_grad = out_grad.copy().reshape(input.shape)
        return in_grad


class matmul(operator):
    def __init__(self):
        super(matmul, self).__init__()

    def forward(self, input, weights):
        """
        # Arguments
            input: numpy array with shape (batch, in_features)
            weights: numpy array with shape (in_features, out_features)

        # Returns
            output: numpy array with shape(batch, out_features)
        """
        return np.matmul(input, weights)

    def backward(self, out_grad, input, weights):
        """
        # Arguments
            out_grad: gradient to the forward output of linear layer, with shape (batch, out_features)
            input: numpy array with shape (batch, in_features)
            weights: numpy array with shape (in_features, out_features)

        # Returns
            in_grad: gradient to the forward input with same shape as input
            w_grad: gradient to weights, with same shape as weights            
        """
        in_grad = np.matmul(out_grad, weights.T)
        w_grad = np.matmul(input.T, out_grad)
        return in_grad, w_grad


class add_bias(operator):
    def __init__(self):
        super(add_bias, self).__init__()

    def forward(self, input, bias):
        '''
        # Arugments
          input: numpy array with shape (batch, in_features)
          bias: numpy array with shape (in_features)

        # Returns
          output: numpy array with shape(batch, in_features)
        '''
        return input + bias.reshape(1, -1)

    def backward(self, out_grad, input, bias):
        """
        # Arguments
            out_grad: gradient to the forward output of linear layer, with shape (batch, out_features)
            input: numpy array with shape (batch, in_features)
            bias: numpy array with shape (out_features)
        # Returns
            in_grad: gradient to the forward input with same shape as input
            b_bias: gradient to bias, with same shape as bias
        """
        in_grad = out_grad
        b_grad = np.sum(out_grad, axis=0)
        return in_grad, b_grad


class linear(operator):
    def __init__(self):
        super(linear, self).__init__()
        self.matmul = matmul()
        self.add_bias = add_bias()

    def forward(self, input, weights, bias):
        """
        # Arguments
            input: numpy array with shape (batch, in_features)
            weights: numpy array with shape (in_features, out_features)
            bias: numpy array with shape (out_features)

        # Returns
            output: numpy array with shape(batch, out_features)
        """
        output = self.matmul.forward(input, weights)
        output = self.add_bias.forward(output, bias)
        # output = np.matmul(input, weights) + bias.reshape(1, -1)
        return output

    def backward(self, out_grad, input, weights, bias):
        """
        # Arguments
            out_grad: gradient to the forward output of linear layer, with shape (batch, out_features)
            input: numpy array with shape (batch, in_features)
            weights: numpy array with shape (in_features, out_features)
            bias: numpy array with shape (out_features)

        # Returns
            in_grad: gradient to the forward input of linear layer, with same shape as input
            w_grad: gradient to weights, with same shape as weights
            b_bias: gradient to bias, with same shape as bias
        """
        # in_grad = np.matmul(out_grad, weights.T)
        # w_grad = np.matmul(input.T, out_grad)
        # b_grad = np.sum(out_grad, axis=0)
        out_grad, b_grad = self.add_bias.backward(out_grad, input, bias)
        in_grad, w_grad = self.matmul.backward(out_grad, input, weights)
        return in_grad, w_grad, b_grad


class conv(operator):
    def __init__(self, conv_params):
        """
        # Arguments
            conv_params: dictionary, containing these parameters:
                'kernel_h': The height of kernel.
                'kernel_w': The width of kernel.
                'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
                'pad': The total number of 0s to be added along the height (or width) dimension; half of the 0s are added on the top (or left) and half at the bottom (or right). we will only test even numbers.
                'in_channel': The number of input channels.
                'out_channel': The number of output channels.
        """
        super(conv, self).__init__()
        self.conv_params = conv_params

    def forward(self, input, weights, bias):
        """
        # Arguments
            input: numpy array with shape (batch, in_channel, in_height, in_width)
            weights: numpy array with shape (out_channel, in_channel, kernel_h, kernel_w)
            bias: numpy array with shape (out_channel)

        # Returns
            output: numpy array with shape (batch, out_channel, out_height, out_width)
        """
        kernel_h = self.conv_params['kernel_h']  # height of kernel
        kernel_w = self.conv_params['kernel_w']  # width of kernel
        pad = self.conv_params['pad']
        stride = self.conv_params['stride']
        in_channel = self.conv_params['in_channel']
        out_channel = self.conv_params['out_channel']

        batch, in_channel, in_height, in_width = input.shape
        out_height = 1 + (in_height - kernel_h + pad) // stride
        out_width = 1 + (in_width - kernel_w + pad) // stride
        output = np.zeros((batch, out_channel, out_height, out_width))

        pad_scheme = (pad//2, pad - pad//2)
        input_pad = np.pad(input, pad_width=((0,0), (0,0), pad_scheme, pad_scheme),
                           mode='constant', constant_values=0)

        # get initial nodes of receptive fields in height and width direction
        recep_fields_h = [stride*i for i in range(out_height)]
        recep_fields_w = [stride*i for i in range(out_width)]
        input_conv = img2col(input_pad, recep_fields_h,
                             recep_fields_w, kernel_h, kernel_w)
        output = np.stack(map(
            lambda x: np.matmul(weights.reshape(out_channel, -1), x) + bias.reshape(-1, 1), input_conv), axis=0)
        
        output = output.reshape(batch, out_channel, out_height, out_width)
        return output

    def backward(self, out_grad, input, weights, bias):
        """
        # Arguments
            out_grad: gradient to the forward output of conv layer, with shape (batch, out_channel, out_height, out_width)
            input: numpy array with shape (batch, in_channel, in_height, in_width)
            weights: numpy array with shape (out_channel, in_channel, kernel_h, kernel_w)
            bias: numpy array with shape (out_channel)

        # Returns
            in_grad: gradient to the forward input of conv layer, with same shape as input
            w_grad: gradient to weights, with same shape as weights
            b_bias: gradient to bias, with same shape as bias
        """
        kernel_h = self.conv_params['kernel_h']  # height of kernel
        kernel_w = self.conv_params['kernel_w']  # width of kernel
        pad = self.conv_params['pad']
        stride = self.conv_params['stride']
        in_channel = self.conv_params['in_channel']
        out_channel = self.conv_params['out_channel']

        batch, in_channel, in_height, in_width = input.shape
        out_height = 1 + (in_height - kernel_h + pad) // stride
        out_width = 1 + (in_width - kernel_w + pad) // stride

        pad_scheme = (pad//2, pad - pad//2)
        input_pad = np.pad(input, pad_width=((0,0), (0,0), pad_scheme, pad_scheme),
                           mode='constant', constant_values=0)
                           
        # get initial nodes of receptive fields in height and width direction
        recep_fields_h = [stride*i for i in range(out_height)]
        recep_fields_w = [stride*i for i in range(out_width)]

        #################### To do ####################
        input_conv = img2col(input_pad, recep_fields_h, recep_fields_w, kernel_h, kernel_w)
        dX = np.stack(map(
            lambda x: np.matmul(weights.reshape(out_channel, -1).T, x), out_grad.reshape(batch, out_channel, -1)), axis=0)
                
        dX_pad = np.zeros(input_pad.shape)
        col = 0
        for h in recep_fields_h:
            for w in recep_fields_w:
                block = dX[:, :, col].reshape(batch, in_channel, kernel_h, kernel_w)
                dX_pad[:, :, h : h + kernel_h, w: w + kernel_w] += block
                col+=1
                
        in_grad = dX_pad[:, :, pad_scheme[0]: pad_scheme[0] + in_height, pad_scheme[0]:pad_scheme[0] + in_width] 
           
        w_grad = np.stack(map(lambda x, y: np.matmul(x.reshape(out_channel, -1), y.T).reshape(weights.shape), out_grad, input_conv), axis=0)
        w_grad = np.sum(w_grad, axis = 0)
        
        b_grad = np.stack(map(lambda x: np.sum(x.reshape(out_channel, -1), axis = 1), out_grad), axis=0)
        b_grad = np.sum(b_grad, axis = 0)
        ###############################################

        return in_grad, w_grad, b_grad


class pool(operator):
    def __init__(self, pool_params):
        """
        # Arguments
            pool_params: dictionary, containing these parameters:
                'pool_type': The type of pooling, 'max' or 'avg'
                'pool_h': The height of pooling kernel.
                'pool_w': The width of pooling kernel.
                'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
                'pad': The total number of 0s to be added along the height (or width) dimension; half of the 0s are added on the top (or left) and half at the bottom (or right). we will only test even numbers.
        """
        super(pool, self).__init__()
        self.pool_params = pool_params

    def forward(self, input):
        """
        # Arguments
            input: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            output: numpy array with shape (batch, in_channel, out_height, out_width)
        """
        pool_type = self.pool_params['pool_type']
        pool_height = self.pool_params['pool_height']
        pool_width = self.pool_params['pool_width']
        stride = self.pool_params['stride']
        pad = self.pool_params['pad']

        batch, in_channel, in_height, in_width = input.shape
        out_height = 1 + (in_height - pool_height + pad) // stride
        out_width = 1 + (in_width - pool_width + pad) // stride

        pad_scheme = (pad//2, pad - pad//2)
        input_pad = np.pad(input, pad_width=((0,0), (0,0), pad_scheme, pad_scheme),
                           mode='constant', constant_values=0)

        recep_fields_h = [stride*i for i in range(out_height)]
        recep_fields_w = [stride*i for i in range(out_width)]

        #################### To do ####################
        output = None
        pool_input = img2col(input_pad, recep_fields_h, recep_fields_w, pool_height, pool_width)
        pool_input = pool_input.reshape(batch, in_channel, -1, out_height, out_width)
        if pool_type == 'max':
            output = np.max(pool_input, axis=2)
        elif pool_type == 'avg':
            output = np.mean(pool_input, axis=2)
        else:
            raise ValueError('Doesn\'t support \'%s\' pooling.' % pool_type)
        ###############################################
        return output

    def backward(self, out_grad, input):
        """
        # Arguments
            out_grad: gradient to the forward output of conv layer, with shape (batch, in_channel, out_height, out_width)
            input: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            in_grad: gradient to the forward input of pool layer, with same shape as input
        """
        pool_type = self.pool_params['pool_type']
        pool_height = self.pool_params['pool_height']
        pool_width = self.pool_params['pool_width']
        stride = self.pool_params['stride']
        pad = self.pool_params['pad']

        batch, in_channel, in_height, in_width = input.shape
        out_height = 1 + (in_height - pool_height + pad) // stride
        out_width = 1 + (in_width - pool_width + pad) // stride

        pad_scheme = (pad//2, pad - pad//2)
        input_pad = np.pad(input, pad_width=((0,0), (0,0), pad_scheme, pad_scheme),
                           mode='constant', constant_values=0)

        recep_fields_h = [stride*i for i in range(out_height)]
        recep_fields_w = [stride*i for i in range(out_width)]

        input_pool = img2col(input_pad, recep_fields_h,
                             recep_fields_w, pool_height, pool_width)
        input_pool = input_pool.reshape(
            batch, in_channel, -1, out_height, out_width)

        if pool_type == 'max':
            input_pool_grad = (input_pool == np.max(input_pool, axis=2, keepdims=True)) * \
                out_grad[:, :, np.newaxis, :, :]

        elif pool_type == 'avg':
            scale = 1 / (pool_height*pool_width)
            input_pool_grad = scale * \
                np.repeat(out_grad[:, :, np.newaxis, :, :],
                          pool_height*pool_width, axis=2)

        input_pool_grad = input_pool_grad.reshape(
            batch, in_channel, -1, out_height*out_width)

        input_pad_grad = np.zeros(input_pad.shape)
        idx = 0
        for i in recep_fields_h:
            for j in recep_fields_w:
                input_pad_grad[:, :, i:i+pool_height, j:j+pool_width] += \
                    input_pool_grad[:, :, :, idx].reshape(
                        batch, in_channel, pool_height, pool_width)
                idx += 1
        in_grad = input_pad_grad[:, :, pad:pad+in_height, pad:pad+in_width]
        return in_grad


class dropout(operator):
    def __init__(self, rate, training=True, seed=None):
        """
        # Arguments
            rate: float[0, 1], the probability of setting a neuron to zero
            training: boolean, apply this layer for training or not. If for training, randomly drop neurons, else DO NOT drop any neurons
            seed: int, random seed to sample from input, so as to get mask, which is convenient to check gradients. But for real training, it should be None to make sure to randomly drop neurons
            mask: the mask with value 0 or 1, corresponding to drop neurons (0) or not (1). same shape as input
        """
        self.rate = rate
        self.seed = seed
        self.training = training
        self.mask = None

    def forward(self, input):
        """
        # Arguments
            input: numpy array with any shape

        # Returns
            output: same shape as input
        """
        if self.training:
            scale = 1/(1-self.rate)
            np.random.seed(self.seed)
            p = np.random.random_sample(input.shape)
            # Please use p as the probability to decide whether drop or not
            self.mask = (p >= self.rate).astype('int')
            output = input * self.mask * scale
        else:
            output = input
        return output

    def backward(self, out_grad, input):
        """
        # Arguments
            out_grad: gradient to forward output of dropout, same shape as input
            input: numpy array with any shape
            mask: the mask with value 0 or 1, corresponding to drop neurons (0) or not (1). same shape as input

        # Returns
            in_grad: gradient to forward input of dropout, same shape as input
        """
        if self.training:
            scale = 1/(1-self.rate)
            in_grad = scale * self.mask * out_grad
        else:
            in_grad = out_grad
        return in_grad


class vanilla_rnn(operator):
    def __init__(self):
        super(vanilla_rnn, self).__init__()

    def forward(self, input, kernel, recurrent_kernel, bias):
        """
        # Arguments
            inputs: [input numpy array with shape (batch, in_features), 
                    state numpy array with shape (batch, units)]

        # Returns
            outputs: numpy array with shape (batch, units)
        """
        x, prev_h = input
        output = np.tanh(x.dot(kernel) + prev_h.dot(recurrent_kernel) + bias)
        return output

    def backward(self, out_grad, input, kernel, recurrent_kernel, bias):
        """
        # Arguments
            out_grad: numpy array with shape (batch, units), gradients to outputs
            inputs: [input numpy array with shape (batch, in_features), 
                    state numpy array with shape (batch, units)], same with forward inputs

        # Returns
            in_grad: [gradients to input numpy array with shape (batch, in_features), 
                        gradients to state numpy array with shape (batch, units)]
        """
        x, prev_h = input
        tanh_grad = np.nan_to_num(
            out_grad*(1-np.square(self.forward(input, kernel, recurrent_kernel, bias))))

        in_grad = [np.matmul(tanh_grad, kernel.T), np.matmul(
            tanh_grad, recurrent_kernel.T)]
        kernel_grad = np.matmul(np.nan_to_num(x.T), tanh_grad)
        r_kernel_grad = np.matmul(np.nan_to_num(prev_h.T), tanh_grad)
        b_grad = np.sum(tanh_grad, axis=0)

        return in_grad, kernel_grad, r_kernel_grad, b_grad

class lstm(operator):
    def __init__(self):
        super(lstm, self).__init__()

    def forward(self, input, kernel, recurrent_kernel):
        """
        # Arguments
            inputs: [input numpy array with shape (batch, in_features), 
                    cell state numpy array with shape (batch, units),
                    hidden state numpy array with shape (batch, units)]

        # Returns
            outputs: [New hidden state numpy array with shape (batch, units),
                      New cell_state numpy array with shape (batch, units)]
                    
        Note: We assume no bias term in lstm           
        """
        x, prev_c, prev_h = input #prev_c: previous cell state; prev_h: previous hidden state
        _, all_units = kernel.shape
        units = all_units // 4
        kernel_i, kernel_f, kernel_c, kernel_o= kernel[:, :units], kernel[:, units:2*units], kernel[:, 2*units:3*units], kernel[:, 3*units:all_units],
        recurrent_kernel_i = recurrent_kernel[:, :units] # recurrent weight of input gate
        recurrent_kernel_f = recurrent_kernel[:, units:2*units] # recurrent weight of forget gate
        recurrent_kernel_c = recurrent_kernel[:, 2*units:3*units] # recurrent weight of cell gate
        recurrent_kernel_o = recurrent_kernel[:, 3*units:all_units] # recurrent weight of output gate

        #################### To do ####################
        x_f = sigmoid(x.dot(kernel_f) + prev_h.dot(recurrent_kernel_f))
        x_i = sigmoid(x.dot(kernel_i) + prev_h.dot(recurrent_kernel_i))
        x_o = sigmoid(x.dot(kernel_o) + prev_h.dot(recurrent_kernel_o))
        x_c = np.tanh(x.dot(kernel_c) + prev_h.dot(recurrent_kernel_c))
        cell = x_f*prev_c + x_i*x_c
        hidden = x_o*np.tanh(cell)
        ###############################################   

        return hidden, cell

    def backward(self, out_grad, input, kernel, recurrent_kernel):
        """
        # Arguments
            out_grad: [gradient to output_hidden state, gradient to output_cell_state]
            inputs: [input numpy array with shape (batch, in_features), 
                    cell state numpy array with shape (batch, units),
                    hidden state numpy array with shape (batch, units)]

        # Returns
            in_grad: [gradients to input numpy array with shape (batch, in_features),
                        gradients to cell state numpy array with shape (batch, units),
                        gradients to hidden state numpy array with shape (batch, units)]
        """
        x, prev_c, prev_h = input #prev_c: previous cell state; prev_h: previous hidden state
        _, all_units = kernel.shape
        units = all_units // 4
        kernel_i, kernel_f, kernel_c, kernel_o= kernel[:, :units], kernel[:, units:2*units], kernel[:, 2*units:3*units], kernel[:, 3*units:all_units],
        recurrent_kernel_i = recurrent_kernel[:, :units]
        recurrent_kernel_f = recurrent_kernel[:, units:2*units]
        recurrent_kernel_c = recurrent_kernel[:, 2*units:3*units]
        recurrent_kernel_o = recurrent_kernel[:, 3*units:all_units]
        h_grad, c_grad = out_grad
        x_f = sigmoid(x.dot(kernel_f) + prev_h.dot(recurrent_kernel_f))
        x_i = sigmoid(x.dot(kernel_i) + prev_h.dot(recurrent_kernel_i))
        x_o = sigmoid(x.dot(kernel_o) + prev_h.dot(recurrent_kernel_o))
        x_c = np.tanh(x.dot(kernel_c) + prev_h.dot(recurrent_kernel_c))
        c = x_i * x_c + x_f * prev_c
        h = x_o * np.tanh(c)
        do = h_grad * np.tanh(c)
        df = c_grad * prev_c
        dc = c_grad * x_i
        di = c_grad * x_c
        dAc = dc * (1-x_c*x_c)
        dAi = di * x_i * (1-x_i)
        dAf = df * x_f * (1-x_f)
        dAo = do * x_o * (1-x_o)
        x_grad = dAc.dot(kernel_c.T)+dAi.dot(kernel_i.T)+dAf.dot(kernel_f.T)+dAo.dot(kernel_o.T)
        kernel_c_grad = x.T.dot(dAc)
        kernel_i_grad = x.T.dot(dAi)
        kernel_f_grad = x.T.dot(dAf)
        kernel_o_grad = x.T.dot(dAo)
        prev_h_grad = dAc.dot(recurrent_kernel_c.T)+dAi.dot(recurrent_kernel_i.T)+dAf.dot(recurrent_kernel_f.T)+dAo.dot(recurrent_kernel_o.T)
        recurrent_kernel_c_grad = prev_h.T.dot(dAc)
        recurrent_kernel_i_grad = prev_h.T.dot(dAi)
        recurrent_kernel_f_grad = prev_h.T.dot(dAf)
        recurrent_kernel_o_grad = prev_h.T.dot(dAo)
        prev_c_grad = c_grad * x_f + prev_h_grad * x_o * (1-np.tanh(prev_c) * np.tanh(prev_c))

        in_grad = [x_grad, prev_c_grad, prev_h_grad]
        kernel_grad = np.concatenate([kernel_i_grad, kernel_f_grad, kernel_c_grad,kernel_o_grad], axis=-1)
        recurrent_kernel_grad = np.concatenate([recurrent_kernel_i_grad, recurrent_kernel_f_grad, recurrent_kernel_c_grad,recurrent_kernel_o_grad], axis=-1)

        return in_grad, kernel_grad, recurrent_kernel_grad

        
class softmax_cross_entropy(operator):
    def __init__(self):
        super(softmax_cross_entropy, self).__init__()

    def forward(self, input, labels):
        """
        # Arguments
            input: numpy array with shape (batch, num_class)
            labels: numpy array with shape (batch,)
            eps: float, precision to avoid overflow

        # Returns
            output: scalar, average loss
            probs: the probability of each category
        """
        # precision to avoid overflow
        eps = 1e-12

        batch = len(labels)
        input_shift = input - np.max(input, axis=1, keepdims=True)
        Z = np.sum(np.exp(input_shift), axis=1, keepdims=True)

        log_probs = input_shift - np.log(Z+eps)
        probs = np.exp(log_probs)
        output = -1 * np.sum(log_probs[np.arange(batch), labels]) / batch
        return output, probs

    def backward(self, input, labels):
        """
        # Arguments
            input: numpy array with shape (batch, num_class)
            labels: numpy array with shape (batch,)
            eps: float, precision to avoid overflow

        # Returns
            in_grad: gradient to forward input of softmax cross entropy, with shape (batch, num_class)
        """
        # precision to avoid overflow
        eps = 1e-12

        batch = len(labels)
        input_shift = input - np.max(input, axis=1, keepdims=True)
        Z = np.sum(np.exp(input_shift), axis=1, keepdims=True)
        log_probs = input_shift - np.log(Z+eps)
        probs = np.exp(log_probs)

        in_grad = probs.copy()
        in_grad[np.arange(batch), labels] -= 1
        in_grad /= batch
        return in_grad

        