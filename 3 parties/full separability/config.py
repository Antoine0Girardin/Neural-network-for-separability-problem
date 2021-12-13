import numpy as np
import pickle
import os
import tensorflow as tf
import qutip as q
from targets import isotropic_state, GHZ, W_state


class Config:
    """ Config is just a collection of all metadata which can be accessed from auxiliary files as well. """
    def __init__(self):
        
        
        self.dim_a = 2
        self.dim_b = 2
        self.dim_c = 2
        
        self.inputsize=20
        self.a_outputsize = (2*self.dim_a-1)+(2*self.dim_b-1)+(2*self.dim_c-1)+1 ## 2n-2 for each pure states, +1 for p(lambda) 

        # Neural network parameters
        self.latin_depth = 1
        self.latin_width = 100

        # Training procedure parameters
        self.batch_size = self.inputsize
        self.no_of_batches = 3000 # How many batches to go through during training.
        self.weight_init_scaling = 10 #2.#10. # default is 1. Set to larger values to get more variance in initial weights.
        self.optimizer = 'adadelta' # #'Adam'sgd '' #'Adagrad'
        self.lr =  .5 
        self.decay = 0.001/2
        self.momentum = 0.25
        self.loss = 'td'    #'td' for trace distance, 'qre' for quantum relative entropy

        # Neural network parameters that I don't change much
        self.no_of_validation_batches = 100 # How many batches to go through in each validation step. If batch size is large, 1 should be enough.
        self.change_batch_size(self.batch_size) #updates test batch size
        self.greek_depth = 0 # set to 0 if trivial neural networks at sources
        self.greek_width = 1
        self.activ = 'relu' # activation for most of NN
        self.activ2 = 'sigmoid' # activation for last dense layer
        self.kernel_reg = None
    
        
    def change_dmtargetGHZ(self, a):
        self.dmtarget = a*GHZ(8,1) + (1-a)*tf.cast(tf.eye(8)/8, 'complex64')
        
    def change_dmtargetW(self, a):
        self.dmtarget = W_state(a)

    def change_batch_size(self,new_batch_size):
        self.batch_size = new_batch_size
        self.batch_size_test = int(self.no_of_validation_batches*self.batch_size) # in case we update batch_size we should also update the test batch size

    def save(self,name):
        with open('./saved_configs/'+name, 'wb') as f:
            pickle.dump(self, f)


def load_config(name):
    with open('./saved_configs/'+name, 'rb') as f:
        temp = pickle.load(f)
    return temp

def initialize():
    """ Initializes a Config class as a global variable pnn (for Parameters of Neural Network).
    The pnn object should be accessible and modifiable from all auxiliary files.
    """
    global pnn
    pnn = Config()
    pnn.save('initial_pnn')
