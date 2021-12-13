import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda
import tensorflow.compat.v1 as tfv1
from scipy.stats import entropy
from tensorflow.keras.initializers import VarianceScaling
import time
import qutip as q

import config as cf

cf.initialize()
n_a=cf.pnn.dim_a
n_b=cf.pnn.dim_b

def build_model():
    """ Build NN """
    inputTensor = Input((cf.pnn.inputsize,))

     ## Note: increasing the variance of the initialization seemed to help in some cases, especially when the number if outputs per party is 4 or more.
    kernel_init = VarianceScaling(scale=cf.pnn.weight_init_scaling, mode='fan_in', distribution='truncated_normal', seed=None)
    
    group_a = Dense(cf.pnn.latin_width,activation=cf.pnn.activ, kernel_regularizer=cf.pnn.kernel_reg, kernel_initializer = kernel_init)(inputTensor)
 
    for _ in range(cf.pnn.latin_depth-1):
        group_a = Dense(cf.pnn.latin_width,activation=cf.pnn.activ, kernel_regularizer=cf.pnn.kernel_reg, kernel_initializer = kernel_init)(group_a)

    group_a = Dense(cf.pnn.a_outputsize,activation=cf.pnn.activ2, kernel_regularizer=cf.pnn.kernel_reg)(group_a)

    outputTensor = group_a

    model = Model(inputTensor,outputTensor)
    return model

def trace_distance(p,q): 
    e,_=tf.linalg.eigh(p-q)
    return  0.5*tf.math.reduce_sum(tf.math.abs(e),axis=-1)

def keras_distance(p,q):
    """ Distance used in loss function.
    Implemented losses:
        td: trace distance
    """
    if cf.pnn.loss.lower() == 'td':       
        e,_=tf.linalg.eigh(p-q)
        return  0.5*tf.math.reduce_sum(tf.math.abs(e),axis=-1) 
    
    else:
        print('define the pnn.loss')
    

def densitynK(arg, d):        #--- len(arg)=2n-1 ## d is the dimension of the state we build here
    """make a density matrix out of 2n-1 numbers (pure state) """
    arg=2*arg-1
    norm= tf.sqrt(tf.reduce_sum(tf.square(arg)))
    for i in range(d):
        if i == 0:
            temp = tf.slice(arg, (i,), (1,))
            temp/=norm
            pure = tf.cast(temp, tf.complex64)

        else:
            temp1 = tf.slice(arg, (2*i-1,), (1,))
            temp2 = tf.slice(arg, (2*i,), (1,))
            temp1/=norm
            temp2/=norm
            pure = tf.concat((pure, tf.dtypes.complex(temp1, temp2)), axis =-1)
            
    pure=tf.reshape(pure, (d,1))
    
    return tf.matmul(pure, pure, adjoint_b=True)


def integral(r_ABC, p1):
    """compute the integral with multiple matrices"""
    s=0
    for i in range(r_ABC.shape[0]):
        s+= p1[i]*r_ABC[i] 
    return s

def kron_prod(r_a,r_b):
    """kronecker product of two matrices"""
    kro=tf.linalg.LinearOperatorKronecker([tf.linalg.LinearOperatorFullMatrix(r_a),tf.linalg.LinearOperatorFullMatrix(r_b)])
    return kro.to_dense()

def reorder(R): 
    """reorder from bac basis to abc (inverse to first two indices)"""

    R=tf.reshape(R, (n_a*n_b, n_a*n_b))
    for i in range(16):
        if i==0:
            m = tf.slice(R, (0,0), (2,2))
            m=tf.reshape(m, (1,2,2))
        if i>0 and i<=3:
            m=tf.concat([m, tf.reshape(tf.slice(R, (0,2*i), (2,2)), (1,2,2))], 0)
        if i>3 and i<=7:
            m=tf.concat([m, tf.reshape(tf.slice(R, (2,2*(i-4)), (2,2)), (1,2,2))], 0)
        if i>7 and i<=11:
            m=tf.concat([m, tf.reshape(tf.slice(R, (4,2*(i-8)), (2,2)), (1,2,2))], 0)
        if i>11 and i<=15:
            m=tf.concat([m, tf.reshape(tf.slice(R, (6,2*(i-12)), (2,2)), (1,2,2))], 0)

    line12 = tf.concat([m[0], m[2], m[1], m[3]], -1)
    line34 = tf.concat([m[8], m[10], m[9], m[11]], -1)
    line56 = tf.concat([m[4], m[6], m[5], m[7]], -1)  
    line78 = tf.concat([m[12], m[14], m[13], m[15]], -1) 

    R2 = tf.concat([line12, line34, line56, line78], 0)          
    return R2

def strategy(y_pred):
    """return the matrices r_a and r_b with weights"""
    l_a = 2*n_a-1
    l_b = 2*n_b-1
    
    a=y_pred[:, :l_a]              #len(y_pred) = batch_size

    bc=y_pred[:, l_a:l_a+l_b]
    
    p1=y_pred[:, l_a+l_b:l_a+l_b+1] 
 
    
    p1=tf.cast(p1, 'complex64')
                  
    weight1=p1/tf.math.reduce_sum(p1)  #weight1 sum to one
    
    for i in range(cf.pnn.batch_size):
        if i == 0:
            r_a = densitynK(tf.reshape(tf.slice(a, (0,0), (1,(2*n_a-1))),((2*n_a-1),)), n_a)
            r_bc = densitynK(tf.reshape(tf.slice(bc, (0,0), (1,(2*n_b-1))),((2*n_b-1),)), n_b)
        else:
            r_a = tf.concat([r_a, densitynK(tf.reshape(tf.slice(a, (i,0), (1,(2*n_a-1))), ((2*n_a-1),)), n_a)], axis=0)     
            r_bc = tf.concat([r_bc, densitynK(tf.reshape(tf.slice(bc, (i,0), (1,(2*n_b-1))), ((2*n_b-1),)), n_b)], axis=0)


    r_a=tf.reshape(r_a, (cf.pnn.batch_size,n_a,n_a))
    
    r_bc=tf.reshape(r_bc, (cf.pnn.batch_size,n_b,n_b))
    
    r_ABC = kron_prod(r_a, r_bc)
    return r_ABC
    
    
def customLoss_matrix(y_pred):          ##y_pred is what the nn gives out size (batch_size, output size =6 (3 for each matrix))
    """make the loss out of the output of the nn"""
    l_a = 2*n_a-1
    l_b = 2*n_b-1
    
    # tf.print(y_pred)
    
    a=y_pred[:, :l_a]              #len(y_pred) = batch_size

    bc=y_pred[:, l_a:l_a+l_b]
    
    p1=y_pred[:, l_a+l_b:l_a+l_b+1] 
 
    
    p1=tf.cast(p1, 'complex64')
                  
    weight1=p1/tf.math.reduce_sum(p1)  #weight1 sum to one
    
    for i in range(cf.pnn.batch_size):
        if i == 0:
            r_a = densitynK(tf.reshape(tf.slice(a, (0,0), (1,(2*n_a-1))),((2*n_a-1),)), n_a)
            r_bc = densitynK(tf.reshape(tf.slice(bc, (0,0), (1,(2*n_b-1))),((2*n_b-1),)), n_b)
        else:
            r_a = tf.concat([r_a, densitynK(tf.reshape(tf.slice(a, (i,0), (1,(2*n_a-1))), ((2*n_a-1),)), n_a)], axis=0)     
            r_bc = tf.concat([r_bc, densitynK(tf.reshape(tf.slice(bc, (i,0), (1,(2*n_b-1))), ((2*n_b-1),)), n_b)], axis=0)


    r_a=tf.reshape(r_a, (cf.pnn.batch_size,n_a,n_a))
    
    r_bc=tf.reshape(r_bc, (cf.pnn.batch_size,n_b,n_b))
    
    r_ABC = kron_prod(r_a, r_bc)
    
    return integral(r_ABC, weight1)


def customLoss_2qubits(y_true, y_pred):
    return keras_distance(tf.cast(tf.reshape(y_true[0,:],(n_a*n_b,n_a*n_b)), 'complex64'), customLoss_matrix(y_pred))

def customLoss_eval(y_pred):
    return keras_distance(tf.cast(cf.pnn.dmtarget, 'complex64'), customLoss_matrix(y_pred))

def metric_td(y_true, y_pred):
    return trace_distance(tf.cast(tf.reshape(y_true[0,:],(n_a*n_b,n_a*n_b)), 'complex64'), customLoss_matrix(y_pred))

     
def generate_xy_batch():  #the inputs are define at the begining of the code
    while True:
        x_train = np.array([i for i in range(cf.pnn.inputsize)])
        x_one_hot = tf.one_hot(x_train, depth=cf.pnn.inputsize)         
        y_true = np.array([np.array(cf.pnn.dmtarget).reshape(((n_a*n_b)**2,)) for _ in range(cf.pnn.batch_size)])
        yield (x_one_hot, y_true)     
        
def generate_x_test():
    while True:
        x_train = np.array([i for i in range(cf.pnn.inputsize)])
        x_one_hot = tf.one_hot(x_train, depth=cf.pnn.inputsize)      
        yield x_one_hot 
        
        
def CC_mindelta():
  return tf.keras.callbacks.EarlyStopping(monitor='metric_td', min_delta=0.0002, patience=1, verbose=0, mode='auto',baseline=None, restore_best_weights=False)

class CC_minloss(tf.keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs=None):
        if logs.get('val_metric_td') <= 0.002:
            tf.print('\n\n\n-----> stoping the training because the trace distance is smaller than 0.002')
            self.model.stop_training = True

def single_run():
    """ Runs training algorithm for a single target distribution. Returns model."""
    # Model and optimizer related setup.
    
    K.clear_session()
    model = build_model()

    if cf.pnn.optimizer.lower() == 'adadelta':
        optimizer = tf.keras.optimizers.Adadelta(lr=cf.pnn.lr, rho=0.95, epsilon=1e-07, decay=cf.pnn.decay)
    elif cf.pnn.optimizer.lower() == 'sgd':
        optimizer = tf.keras.optimizers.SGD(lr=cf.pnn.lr, decay=cf.pnn.decay, momentum=cf.pnn.momentum, nesterov=True)
    else:
        optimizer = tf.keras.optimizers.SGD(lr=cf.pnn.lr, decay=cf.pnn.decay, momentum=cf.pnn.momentum, nesterov=True)
        print("\n\nWARNING!!! Optimizer {} not recognized. Please implement it if you want to use it. Using SGD instead.\n\n".format(cf.pnn.optimizer))
        cf.pnn.optimizer = 'sgd' # set it for consistency.

    model.compile(loss=customLoss_2qubits, optimizer = optimizer, metrics=[metric_td])  
    
    # Fit model   ## y will be the target
    fit=model.fit(generate_xy_batch(), steps_per_epoch=cf.pnn.no_of_batches, epochs=20, callbacks = [CC_mindelta(), CC_minloss()], verbose=1, validation_data=generate_xy_batch(), validation_steps=cf.pnn.no_of_validation_batches, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=False, initial_epoch=0)
    return model, fit


def single_evaluation(model):
    """ Evaluates the model and returns the resulting distribution as a numpy array. """
    test_pred = model.predict(generate_x_test(), steps=1, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
    r_a,r_b,weight = K.eval(strategy(test_pred))
    
    result = K.eval(customLoss_matrix(test_pred))
    return test_pred,r_a, r_b, weight, result 

def single_evaluation_loss(model):
    """ Evaluates the model and returns the loss. """
    test_pred = model.predict(generate_x_test(), steps=1, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
    td = K.eval(trace_distance(customLoss_matrix(test_pred), cf.pnn.dmtarget))
    result = K.eval(customLoss_eval(test_pred))
    return result, td

def plot_loss(model, loss, t0):
    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    ax = plt.gca()
    plt.text(.9,.9, 'Loss ={}'.format(loss),horizontalalignment='right',verticalalignment='top',transform = ax.transAxes)
    plt.text(.9,.8, 'Time ={}'.format(time.time()-t0),horizontalalignment='right',verticalalignment='top',transform = ax.transAxes)
    #plt.ylim((0,1))
    plt.savefig("./figs/loss.png")
    plt.show()

        
def ppt(a):
    s2 = q.Qobj(q.Qobj(np.array(a)).data.toarray().reshape((n_a*n_b,n_a*n_b)),dims=[[n_a,n_a],[n_b,n_b]])
    return q.partial_transpose(s2, [0,1], method="dense")
    
def test_separability(a):
    e,_=tf.linalg.eigh(np.array(ppt(a)))
    return e[0]

def secondlowesteigPPT(a):
    e,_=tf.linalg.eigh(np.array(ppt(a)))
    return e[1]
    
    
    
    