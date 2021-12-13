import numpy as np
import tensorflow as tf
import qutip as q


#isotropic state for qutrits
def isotropic_state(d,a):

    identity = tf.constant(np.identity(d**2), dtype='complex64')
    
    phi_dm=np.zeros((d**2,d**2))
    
    for i in range(d**2):
        for j in range(d**2):
            if i%(d+1) == 0 and j%(d+1) == 0:
                phi_dm[i][j] = 1/d

    phi_dm = tf.cast(phi_dm,dtype ='complex64')
    
    return ((1-a)/d**2)*identity + a*phi_dm
    
def Horodecki(a):
    d=3
    phi_dm=np.zeros((d**2,d**2))
    
    for i in range(d**2):
        for j in range(d**2):
            if i%(d+1) == 0 and j%(d+1) == 0:
                phi_dm[i][j] = a
            if i==j:
                if i==6 or i ==8:
                    phi_dm[i][j] = (1+a)/2
                else:
                    phi_dm[i][j] = a
            if i==8 and j==6:
                phi_dm[i][j] = tf.math.sqrt(1-a**2)/2
                phi_dm[j][i] = tf.math.sqrt(1-a**2)/2

    phi_dm = tf.cast(phi_dm,dtype ='complex64')
    
    return  phi_dm/(8*a+1)

def GHZ(n,m):
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