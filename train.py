import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU') #uncomment to run on CPU (faster on CPU for low dimension, may be faster at high dimensions on GPU)

import pickle
import os
import matplotlib.pyplot as plt
import time

import config as cf

from utils_nn import single_run, single_evaluation, plot_loss, single_evaluation_loss, test_separability

np.set_printoptions(2, suppress=True)
np.set_printoptions(linewidth=1000)

def single_sweep_training():
    """ train the target once with the config values given in config.py """
    
    #choose the target state (change it in the config.py)
    cf.pnn.change_dmtarget(.0)
    
    # Run and evaluate model
    model,fit = single_run()

    print("time=",time.time()-t0)
    
    result = single_evaluation(model)
    loss, td = single_evaluation_loss(model)
    print(loss)
    print('Trace distance =', td)
    print(result)
    
    print('target = ', cf.pnn.dmtarget)

    # Plot distances
    plot_loss(fit, loss, td, t0)
    return loss, result
    

if __name__ == '__main__':
    #Create directories for saving stuff
    for dir in ['figs']:
        if not os.path.exists(dir):
            os.makedirs(dir)
    # Set up the Parameters of the Neural Network (i.e. the config object)
    t0=time.time()
    cf.initialize()
    n=cf.pnn.dim

    # Run single sweep
    loss, result = single_sweep_training()
