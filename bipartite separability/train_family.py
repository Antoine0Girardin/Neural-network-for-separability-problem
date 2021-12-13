import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU') #uncomment to run on CPU (faster on CPU for low dimension, may be faster at high dimensions on GPU)

import pickle
import os
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit

import config as cf

from utils_nn import single_run, single_evaluation, plot_loss, single_evaluation_loss, test_separability

np.set_printoptions(2, suppress=True)

def single_sweep_training():
    """ Goes through some targets defined in config.py and return the distance to the constructed state"""
    plot_a=[]
    plot_dist=[]
    plot_td=[]
    
    nb_trains=1 #choose how many time to train each target
    
    for a in  (0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1):
        temp_loss=[]
        temp_td=[]
        
        #choose the target family of state (change it in the config.py)
        cf.pnn.change_dmtarget(a)
        
        for trains in range(nb_trains):
            print('Starting for a = {}, trains = {}'.format(a, trains+1))
            
            # Run and evaluate model
            model,fit = single_run()

            print("time=",time.time()-t0)
        
            loss, td = single_evaluation_loss(model)
            
            temp_loss.append(loss)
            temp_td.append(td)
            if td<0.005:
                 break
        
        plot_a.append(a)
        plot_dist.append(min(temp_loss))
        plot_td.append(min(temp_td))
        
    
    return plot_a, plot_dist, plot_td
    

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
    plot_a, plot_dist, plot_td = single_sweep_training()    
    
    # Plot distances    
    # plt.scatter(plot_a, plot_dist, label='loss of the nn')
    plt.scatter(plot_a, plot_td, label='trace distance')
    plt.legend()
    plt.title('Loss of the nn for Werner state in {} dimensions'.format(cf.pnn.dim))
    # plt.title('Loss of the nn for Isotropic state in {} dimensions'.format(cf.pnn.dim))
    # plt.title('Loss of the nn for Mintert 104 state')

    plt.ylabel('loss')
    plt.xlabel('a')
    plt.savefig("./figs/loss.png")
    plt.savefig("./figs/loss.pdf")
    plt.show()