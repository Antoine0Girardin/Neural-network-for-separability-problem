import numpy as np
import pickle
import os
import tensorflow as tf
import qutip as q
from targets import isotropic_state, Mintert, Werner_state

class Config:
    """ Config is just a collection of all metadata which can be accessed from auxiliary files as well. """
    def __init__(self):
        self.dim = 2

        self.inputsize=6 # choose how many pure state for the separable mixed state
        self.a_outputsize = 2*(2*self.dim-1)+1  ## 2n-1 for each pure states, +1 for p(lambda) 

        # Neural network parameters
        self.latin_depth = 1
        self.latin_width = 100

        # Training procedure parameters
        self.batch_size = self.inputsize
        self.no_of_batches = 3000 # How many batches to go through during training.
        self.weight_init_scaling = 2.#10. # default is 1. Set to larger values to get more variance in initial weights.
        self.optimizer = 'adadelta' # #'Adam'  #'Adagrad'''sgd
        self.lr =  0.5
        self.decay = 0.001/2
        self.momentum = 0.25
        self.loss = 'td'    #'td'  'qre'  frobenius

        # Neural network parameters that I don't change much
        self.no_of_validation_batches = 100 # How many batches to go through in each validation step. If batch size is large, 1 should be enough.
        self.change_batch_size(self.batch_size) #updates test batch size

        self.activ = 'relu' # activation for most of NN
        self.activ2 = 'sigmoid' # activation for last dense layer
        self.kernel_reg = None
        self.activity_reg = 0.001  

    def change_dmtarget(self, a): #change here the target state imported from targets.py
        """set the target state to a family defined in targets.py"""
        # self.dmtarget = isotropic_state(self.dim, a)
        self.dmtarget = Werner_state(self.dim, a)
        # self.dmtarget = Mintert(a)

    def change_batch_size(self,new_batch_size):
        self.batch_size = new_batch_size
        self.batch_size_test = int(self.no_of_validation_batches*self.batch_size) # in case we update batch_size we should also update the test batch size


def initialize():
    """ Initializes a Config class as a global variable pnn (for Parameters of Neural Network).
    The pnn object should be accessible and modifiable from all auxiliary files.
    """
    global pnn
    pnn = Config()