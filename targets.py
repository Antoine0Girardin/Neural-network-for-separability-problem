import numpy as np
import tensorflow as tf
import qutip as q

#isotropic state for qutrits
def isotropic_state(d,a):
    """Isotropic states with dimension d, a is the purity of the state"""
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

def Werner_state(d,p):
    """Werner state with dimension d, a is the purity of the state"""
    I = tf.constant(np.identity(d**2), dtype='complex64')
    
    F = np.zeros((d**2, d**2))
    
    for i in range(d):
        for j in range(d):
            F[i+d*j][i*d+j] =1
            
    return p*2/(d*(d+1))*0.5*(I+F) + (1-p)*2/(d*(d-1))*0.5*(I-F)

def Mintert(a): #dim=3, a in [-5/2, 5/2]
    """State defined in the paper following paper at equation 104
    "1.Mintert, F., Carvalho, A. R. R., Kus, M. & Buchleitner, A. Measures and dynamics of entangled states. Physics Reports 415, 207–259 (2005)."
    Note : a is a parameter from -5/2 to 5/2"""

    bp=5/2+a
    bm=5/2-a
    target = 1/21*tf.constant([
        [2., 0., 0., 0., 2., 0., 0., 0., 2.],
        [0., bm, 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., bp, 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., bp, 0., 0., 0., 0., 0.],
        [2., 0., 0., 0., 2., 0., 0., 0., 2.],
        [0., 0., 0., 0., 0., bm, 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., bm, 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., bp, 0.],
        [2., 0., 0., 0., 2., 0., 0., 0., 2.]], dtype='complex64')
    return target

def k_p(r_a,r_b):
    kro=tf.linalg.LinearOperatorKronecker([tf.linalg.LinearOperatorFullMatrix(r_a),tf.linalg.LinearOperatorFullMatrix(r_b)])
    return kro.to_dense()

def Smolin(p):
    """Smolin state, p is the amount of noise"""
    identity = tf.constant(np.identity(16), dtype='complex64')
    sx=tf.constant([[0,1], [1,0]], dtype='complex64')
    sy=tf.constant(np.array([[0,-1j], [1j,0]]), dtype='complex64')
    sz=tf.constant([[1,0], [0,-1]], dtype='complex64')
    r=1/16*(identity+k_p(k_p(sx, sx), k_p(sx, sx))+k_p(k_p(sy, sy), k_p(sy, sy))+k_p(k_p(sz, sz), k_p(sz, sz)))
    return (1-p)*r + p*identity/16

def W_state(a):
    """W state in dimension 3, a is the purity of the state"""
    n = 3
    W = np.zeros((n,2**n,1), dtype = 'complex64')
    W[0][1] = 1
    W[1][2] = 1
    W[2][4] = 1
    
    Wstate = 1/np.sqrt(n)*np.sum(W, axis=0)
    
    R=tf.matmul(Wstate, Wstate, adjoint_b=True)
    I = tf.constant(np.identity(8), dtype='complex64')
    return a*R + (1-a)*I