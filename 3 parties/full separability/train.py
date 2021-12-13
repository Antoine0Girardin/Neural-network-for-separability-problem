import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit

import config as cf

from utils_nn import single_run, single_evaluation, plot_loss, single_evaluation_loss, test_separability

np.set_printoptions(2, suppress=True)

def fit_function(x,a,b):
    return a*x +b

def single_sweep_training():
    """ Goes through some density matrices in cf.pnn.target_distributions once with the config values given in config.py """
    plot_a=[]
    plot_dist=[]
    plot_td=[]
    
    for a in (0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.):
        # cf.pnn.change_dmtargetGHZ(a)
        cf.pnn.change_dmtargetW(a)
        
        print('Starting for a =', a)
        # Run and evaluate model
        model,fit = single_run()
        #result = single_evaluation(model)
        print("time=",time.time()-t0)
        
        loss, td= single_evaluation_loss(model)
        
        plot_a.append(a)
        plot_dist.append(loss)
        plot_td.append(td)
    
    return plot_a, plot_dist, plot_td
    

if __name__ == '__main__':
    #Create directories for saving stuff
    for dir in ['saved_configs', 'figs']:
        if not os.path.exists(dir):
            os.makedirs(dir)
    # Set up the Parameters of the Neural Network (i.e. the config object)
    t0=time.time()
    cf.initialize()
    n_a=cf.pnn.dim_a
    n_b=cf.pnn.dim_b
    n_c=cf.pnn.dim_c
    
    # Run single sweep
    plot_a, plot_dist, plot_td = single_sweep_training()    

    # Plot distances    
    plt.scatter(plot_a, plot_dist, label='Loss of the neural network')
 
    plt.scatter(plot_a, plot_td, label='Trace distance')
    
    plt.legend(loc=0)
    # plt.title('Trace distance for total separability of the nn for GHZ states (3 qubits) with noise')
    plt.title('Trace distance for total separability of the nn for W states (3 qubits) with noise')

    plt.ylabel('loss')
    plt.xlabel('a')
    plt.savefig("./figs/loss.png")
    plt.savefig("./figs/loss.pdf")
    plt.show()
    