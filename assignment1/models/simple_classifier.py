from nn.layers import *
from nn.loss import *

class simple_classifier():
 
    def __init__(self, n_in, n_out1, n_out2):
        """Initialization

        # Arguments
            n_in: the number of input features
            n_out1: the output features of first fully connected layer
            n_out2: the output features of second fully connected layer
            
        # Returns
            new_xs: dictionary, new weights of model
        """
        self.linear1 = Linear(n_in, n_out1)
        self.linear2 = Linear(n_out1, n_out2)
        self.relu = ReLU()
        self.softmax_cross_entropy = SoftmaxCrossEntropy(n_out2)
        
        # cacehs to save intermedia results of forward
        self.caches = []

    def accuracy(self, probs, target):
        # probs: probability that each image is labeled as 1
        # target: ground truth label
        
        prediction = probs.argmax(axis=-1)    
        acc = np.mean(prediction == target)
        return acc * 100


    def get_params(self):
        layer1_params, layer1_grads = self.linear1.get_params('layer-1th')
        layer2_params, layer2_grads = self.linear2.get_params('layer-2th')
        params = {**layer1_params, **layer2_params}
        #print(params)
        grads = {**layer1_grads,**layer2_grads}
        return params,grads

    def forward(self, X, y):
        # compute the accuracy and loss

        caches = [X] # to save intermedia results for backward pass
        
        # layer 1 forward
        out1 = self.linear1.forward(X)
        caches += [out1]
        out1 = self.relu.forward(out1)
        caches += [out1]

        # layer 2 forward
        out2 = self.linear2.forward(out1)
        caches += [out2]

        self.caches = caches

        # loss
        loss, probs = self.softmax_cross_entropy.forward(out2, y)
        acc = self.accuracy(probs, y)  

        return acc, loss

    def backward(self, X, y):

        # loss backward
        inp = self.caches.pop()
        in_grad = self.softmax_cross_entropy.backward(inp, y)
        
        # TODO 
        #layer 2 backward
        inp_l2 = self.caches.pop()
        in_grad_l2 = self.linear2.backward(in_grad, inp_l2)
        
        # TODO 
        #layer 1 backward
        inp_re = self.caches.pop()
        in_grad_re = self.relu.backward(in_grad_l2, inp_re)
        
        inp_l1 = self.caches.pop()
        in_grad_l1 = self.linear1.backward(in_grad_re, inp_l1)
        return in_grad_l1

    
    def update(self, new_param_dict):
        linear1_dict = {}
        linear2_dict = {}
        for k, v in new_param_dict.items():
            if 'layer-1th' in k:
                linear1_dict[k] = v
            elif 'layer-2th' in k:
                linear2_dict[k] = v
        self.linear1.update(linear1_dict)
        self.linear2.update(linear2_dict)

