import numpy as np
import tensorflow as tf

def GHZ(n,m):
    """GHZ state, n and m are the dimension of each state"""
    ghz= np.zeros((n*m,n*m)) 
    ghz[0][0]=.5
    ghz[0][n*m-1]=.5
    ghz[n*m-1][0]=.5
    ghz[n*m-1][n*m-1]=.5
    return tf.cast(ghz,dtype ='complex64')

def W_state(a):
    n = 3
    W = np.zeros((n,2**n,1), dtype = 'complex64')
    W[0][1] = 1
    W[1][2] = 1
    W[2][4] = 1
    
    Wstate = 1/np.sqrt(3)*np.sum(W, axis=0)
    
    R=tf.matmul(Wstate, Wstate, adjoint_b=True)
    I = 1/8*tf.constant(np.identity(8), dtype='complex64')
    return a*R + (1-a)*I
    