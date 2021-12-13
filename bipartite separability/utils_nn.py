import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda
from scipy.stats import entropy
from tensorflow.keras.initializers import VarianceScaling
import time
import qutip as q

import config as cf

cf.initialize()
n=cf.pnn.dim

def build_model(): 
    """ Build NN """
    inputTensor = Input((cf.pnn.inputsize,))

     ## Note: increasing the variance of the initialization seemed to help in some cases, especially when the number if outputs per party is 4 or more.
    kernel_init = VarianceScaling(scale=cf.pnn.weight_init_scaling, mode='fan_in', distribution='truncated_normal', seed=None)
    
    group_a = Dense(cf.pnn.latin_width,activation=cf.pnn.activ, kernel_regularizer=cf.pnn.kernel_reg, kernel_initializer = kernel_init)(inputTensor)
   
    for _ in range(cf.pnn.latin_depth):
        group_a = Dense(cf.pnn.latin_width,activation=cf.pnn.activ, kernel_regularizer=cf.pnn.kernel_reg, 
                        kernel_initializer = kernel_init)(group_a)

    group_a = Dense(cf.pnn.a_outputsize,activation=cf.pnn.activ2, kernel_regularizer=cf.pnn.kernel_reg,kernel_initializer = kernel_init)(group_a)

    outputTensor = group_a

    model = Model(inputTensor,outputTensor)
    return model

def trace_distance(p,q): 
    """trace distance of p and q"""
    e,_=tf.linalg.eigh(p-q)
    return  0.5*tf.math.reduce_sum(tf.math.abs(e),axis=-1) 

def qre(p,q):
    """quantum relative entropy"""
    eigq,_=tf.linalg.eigh(q)
    eigp,_=tf.linalg.eigh(p)
    return tf.math.reduce_sum(eigq*(tf.math.log(eigq)-tf.math.log(eigp)))

def keras_distance(p,q):
    """ Distance used in loss function.
    Implemented losses:
        td: trace distance (recommended)
        d1: difference element-wise
        d2: difference element squared wise
    """
    #some distances defined here are not usable for training
    if cf.pnn.loss.lower() == 'td':       
        e,_=tf.linalg.eigh(p-q)
        return  0.5*tf.math.reduce_sum(tf.math.abs(e),axis=-1)
    if cf.pnn.loss.lower() == 'hs': #Hilbert-Schmidt distance
        return tf.math.sqrt(tf.linalg.trace(tf.linalg.matmul(tf.linalg.adjoint(p-q), p-q)))
    if cf.pnn.loss.lower() == 'bures':    #2(1-sqrt(fid))    
        e, v= tf.linalg.eigh(p)
        srp=tf.linalg.matmul(tf.linalg.matmul(v, tf.linalg.diag(tf.math.sqrt(e))), tf.linalg.adjoint(v))
        e2,_=tf.linalg.eigh(tf.linalg.matmul(tf.linalg.matmul(srp, q), srp))
        srfid= tf.reduce_sum(tf.math.sqrt(e2))
        return  2*(1-srfid)
    if cf.pnn.loss.lower() == 'fid': 
        e, v= tf.linalg.eigh(p)
        srp=tf.linalg.matmul(tf.linalg.matmul(v, tf.linalg.diag(tf.math.sqrt(e))), tf.linalg.adjoint(v))
        e2,_=tf.linalg.eigh(tf.linalg.matmul(tf.linalg.matmul(srp, q), srp))
        srfid= tf.reduce_sum(tf.math.sqrt(e2))
        return -srfid**2
    if cf.pnn.loss.lower() == 'frobenius':
        diff=p-q
        return tf.math.sqrt(tf.linalg.trace(tf.linalg.matmul(diff, tf.transpose(diff, conjugate=True))))
    if cf.pnn.loss.lower()== 'qre':
        eigq,vq=tf.linalg.eigh(q)
        eigp,vp=tf.linalg.eigh(p)
        logq= tf.linalg.matmul(tf.linalg.matmul(vq, tf.linalg.diag(tf.math.log(q))), tf.linalg.adjoint(vq))
        logp= tf.linalg.matmul(tf.linalg.matmul(vp, tf.linalg.diag(tf.math.log(p))), tf.linalg.adjoint(vp))
        return tf.linalg.trace(tf.linalg.matmul(q, logq-logp))
    if cf.pnn.loss.lower() == 'd1':
        return tf.math.reduce_sum(tf.math.reduce_sum((tf.math.abs(p-q)),axis=0))
    if cf.pnn.loss.lower() == 'd2':
        return tf.math.sqrt(tf.math.reduce_sum(tf.math.reduce_sum((tf.math.squared_difference(p,q)))))
    if cf.pnn.loss.lower() == 'dtest':
        return tf.math.reduce_sum(tf.math.reduce_sum((tf.math.abs(p-q))))
    if cf.pnn.loss.lower() == 'tracenorm':
        diff=p-q
        return tf.linalg.trace(tf.math.sqrt(tf.linalg.matmul(diff,tf.transpose(diff, conjugate=True))))
    else:
        print('define the pnn.loss')
    
def densitynK(arg):
    """make a density matrix out of 2n-1 numbers (pure state) """
    d = n
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
            temp1/= norm
            temp2/= norm
            pure = tf.concat((pure, tf.dtypes.complex(temp1, temp2)), axis =-1)
            
    pure=tf.reshape(pure, (d,1))
    return tf.matmul(pure, pure, adjoint_b=True)

def integral(r_a,r_b,weight):      #r_a, r_b will have the shape array(batch_size,2,2)
    """compute the integral with multiple matrices r_a, r_b"""
    s=0
    kro=tf.linalg.LinearOperatorKronecker([tf.linalg.LinearOperatorFullMatrix(r_a),tf.linalg.LinearOperatorFullMatrix(r_b)])
    for i in range(kro.shape[0]):
        s+=kro.to_dense()[i]*tf.cast(weight[i], 'complex64')
    return s

def strategy(y_pred):
    """return the matrices r_a and r_b with weights"""
    a=y_pred[:, :(2*n-1)]              #len(y_pred) = batch_size
    b=y_pred[:, (2*n-1):(2*n-1)*2]
    p=y_pred[:, (2*n-1)*2:]                     
    weight=p/tf.math.reduce_sum(p)
    
    for i in range(cf.pnn.batch_size):
        if i == 0:
            r_a = densitynK(tf.reshape(tf.slice(a, (0,0), (1,(2*n-1))),((2*n-1),)))
            r_b = densitynK(tf.reshape(tf.slice(b, (0,0), (1,(2*n-1))),((2*n-1),)))
        else:
            r_a = tf.concat([r_a, densitynK(tf.reshape(tf.slice(a, (i,0), (1,(2*n-1))), ((2*n-1),)))], axis=0)
            r_b = tf.concat([r_b, densitynK(tf.reshape(tf.slice(b, (i,0), (1,(2*n-1))), ((2*n-1),)))], axis=0)
    
    r_a=tf.reshape(r_a, (cf.pnn.batch_size,n,n))  
    r_b=tf.reshape(r_b, (cf.pnn.batch_size,n,n))  
    return r_a,r_b, weight
    
  
def customLoss_matrix(y_pred):          ##y_pred is what the nn gives out
    """return the separable density matrix corresponding to the output of the nn"""
    a=y_pred[:, :(2*n-1)]              #len(y_pred) = batch_size
    b=y_pred[:, (2*n-1):(2*n-1)*2]
    p=y_pred[:, (2*n-1)*2:]                     
    weight=p/tf.math.reduce_sum(p)  
    
    for i in range(cf.pnn.batch_size):
        if i == 0:
            r_a = densitynK(tf.reshape(tf.slice(a, (0,0), (1,(2*n-1))),((2*n-1),)))
            r_b = densitynK(tf.reshape(tf.slice(b, (0,0), (1,(2*n-1))),((2*n-1),)))
        else:
            r_a = tf.concat([r_a, densitynK(tf.reshape(tf.slice(a, (i,0), (1,(2*n-1))), ((2*n-1),)))], axis=0)
            r_b = tf.concat([r_b, densitynK(tf.reshape(tf.slice(b, (i,0), (1,(2*n-1))), ((2*n-1),)))], axis=0)
    
    r_a=tf.reshape(r_a, (cf.pnn.batch_size,n,n))  
    r_b=tf.reshape(r_b, (cf.pnn.batch_size,n,n))  
    return integral(r_a,r_b,weight)                             #I need all the r_a, r_b to compute the integral

def customLoss_2qubits(y_true, y_pred):
    """return the loss of the nn (distance between the separable and target state)"""
    return keras_distance(tf.cast(tf.reshape(y_true[0,:],(n**2,n**2)), 'complex64'), customLoss_matrix(y_pred))

def customLoss_eval(y_pred):
    """loss of a prediction, not use in training loop"""
    return keras_distance(tf.cast(cf.pnn.dmtarget, 'complex64'), customLoss_matrix(y_pred))

def metric_td(y_true, y_pred):
    """trace distance, only usefull if the loss of the nn is not the td"""
    y_true = cf.pnn.dmtarget 
    return trace_distance(y_true, customLoss_matrix(y_pred)) 
     
def generate_xy_batch():  
    while True:
        x_train = np.array([i for i in range(cf.pnn.inputsize)])
        x_one_hot = tf.one_hot(x_train, depth=cf.pnn.inputsize)         
        y_true = np.array([np.array(cf.pnn.dmtarget).reshape((n**4,)) for _ in range(cf.pnn.batch_size)])
        yield (x_one_hot, y_true)     
      
def generate_x_test():
    while True:
        x_train = np.array([i for i in range(cf.pnn.inputsize)])
        x_one_hot = tf.one_hot(x_train, depth=cf.pnn.inputsize)      
        yield x_one_hot 
        
       
def CC_mindelta():
    """stop the training at the end of an epoch if the loss didn't decrease enough"""
    return tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0003, patience=1, verbose=0, mode='auto',baseline=None, restore_best_weights=False)

class CC_minloss(tf.keras.callbacks.Callback):
    """stop the training at the end of an epoch if the loss is smaller than a value"""
    def on_epoch_end(self, batch, logs=None):
        if logs.get('val_metric_td') <= 0.005:
            tf.print('\n\n\n-----> stoping the training because the trace distance is smaller than 0.005') 
            self.model.stop_training = True
            

def single_run():
    """ Runs training algorithm for a single target density matrix. Returns model and data to fit."""
    K.clear_session()
    model = build_model()

    if cf.pnn.optimizer.lower() == 'adadelta':
        optimizer = tf.keras.optimizers.Adadelta(lr=cf.pnn.lr, rho=0.95, epsilon=None, decay=cf.pnn.decay)
    elif cf.pnn.optimizer.lower() == 'sgd':
        optimizer = tf.keras.optimizers.SGD(lr=cf.pnn.lr, decay=cf.pnn.decay, momentum=cf.pnn.momentum, nesterov=True)
    else:
        optimizer = tf.keras.optimizers.SGD(lr=cf.pnn.lr, decay=cf.pnn.decay, momentum=cf.pnn.momentum, nesterov=True)
        print("\n\nWARNING!!! Optimizer {} not recognized. Please implement it if you want to use it. Using SGD instead.\n\n".format(cf.pnn.optimizer))
        cf.pnn.optimizer = 'sgd' # set it for consistency.

    model.compile(loss=customLoss_2qubits, optimizer = optimizer, metrics=[metric_td])  
    
    # Fit model
    fit=model.fit(generate_xy_batch(), steps_per_epoch=cf.pnn.no_of_batches, epochs=20, callbacks = [CC_minloss(), CC_mindelta()], verbose=1, validation_data=generate_xy_batch(), validation_steps=cf.pnn.no_of_validation_batches, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=False, initial_epoch=0)
    return model, fit

def single_evaluation(model):
    """ Evaluates the model and returns the resulting density matrix as a numpy array. """
    test_pred = model.predict(generate_x_test(), steps=1, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
    r_a,r_b,weight = K.eval(strategy(test_pred))
    
    result = K.eval(customLoss_matrix(test_pred))
    return test_pred,r_a, r_b, weight, result 

def single_evaluation_loss(model):
    """ Evaluates the model and returns the loss. """
    test_pred = model.predict(generate_x_test(), steps=1, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
    
    result = K.eval(customLoss_eval(test_pred))
    td = K.eval(trace_distance(customLoss_matrix(test_pred), cf.pnn.dmtarget))
    return result, td

def plot_loss(model, loss, td, t0):
    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.plot(model.history['val_metric_td'])
    plt.title('Werner state d={}'.format(cf.pnn.dim))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test', 'trace distance'], loc='upper left')
    ax = plt.gca()
    plt.text(.9,.8, '{} ={}'.format(cf.pnn.loss, loss),horizontalalignment='right',verticalalignment='top',transform = ax.transAxes)
    plt.text(.9,.7, 'Trace distance ={}'.format(td),horizontalalignment='right',verticalalignment='top',transform = ax.transAxes)
    plt.text(.9,.6, 'Time ={}'.format(time.time()-t0),horizontalalignment='right',verticalalignment='top',transform = ax.transAxes)
    #plt.ylim((0,1))
    plt.savefig("./figs/loss.png")
    plt.show()

#some function to study density matrices
def ppt(a):
    """return the partial transpose"""
    s2 = q.Qobj(q.Qobj(np.array(a)).data.toarray().reshape((n**2,n**2)),dims=[[n,n],[n,n]])
    return q.partial_transpose(s2, [0,1], method="dense")
    
def test_separability(a):
    """lowest eigenvalue of the PT"""
    e,_=tf.linalg.eigh(np.array(ppt(a)))
    return e[0]

def secondlowesteigPPT(a):
    """second lowest eigenvalue of the PT"""
    e,_=tf.linalg.eigh(np.array(ppt(a)))
    return e[1]
    
def PT(a):
    """return the partial transpose of a state"""
    s2 = q.Qobj(q.Qobj(np.array(a)).data.toarray().reshape((4,4)),dims=[[2,2],[2,2]])
    return np.array(q.partial_transpose(s2, [0,1], method="dense"))
    
def alleigPPT(a, n):
    """return all eigenvalue of the PT"""
    e,_=tf.linalg.eigh(np.array(ppt(a)))
    return e

def alleigPPT2(a, n):
    """return all eigenvectors of the PT"""
    e, v=tf.linalg.eigh(np.array(ppt(a)))
    return v