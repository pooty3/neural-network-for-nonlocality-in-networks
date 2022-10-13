from distutils.command.build import build
import numpy as np
from itertools import product
import qutip as qt
from scipy.io import loadmat

def target_distribution_gen_all(name, param_range, which_param, other_param):
    """ Generate a set of target distributions by varying one parameter. which_param sets whether distr. param or noise param."""
    if which_param == 1:
        p_target_shapeholder = target_distribution_gen(name, param_range[0], other_param);
    elif which_param == 2:
        p_target_shapeholder = target_distribution_gen(name, other_param, param_range[0]);
    target_distributions = np.ones(param_range.shape + p_target_shapeholder.shape) / (p_target_shapeholder.shape[0])
    for i in range(len(param_range)):
        if which_param == 1:
            p_target = target_distribution_gen(name, param_range[i], other_param);
        elif which_param == 2:
            p_target = target_distribution_gen(name, other_param, param_range[i]);
        target_distributions[i,:] = p_target
    return target_distributions

def target_distribution_gen(name, parameter1, parameter2):
    """ parameter1 is usually a parameter of distribution (not always relevant). parameter2 is usually noise."""
    if name=="CHSH":
        v = parameter2
        p = np.array([
        (-2 + np.sqrt(2) + (-1 + np.sqrt(2))*v)/(16.*(-2 + np.sqrt(2))),(2 + np.sqrt(2) + v + np.sqrt(2)*v)/(32 + 16*np.sqrt(2)),(-2 + np.sqrt(2) + v - np.sqrt(2)*v)/(16.*(-2 + np.sqrt(2))),
        (2 + np.sqrt(2) - (1 + np.sqrt(2))*v)/(16.*(2 + np.sqrt(2))),(2 + np.sqrt(2) + v + np.sqrt(2)*v)/(32 + 16*np.sqrt(2)),(-2 + np.sqrt(2) + (-1 + np.sqrt(2))*v)/(16.*(-2 + np.sqrt(2))),(2 + np.sqrt(2) - (1 + np.sqrt(2))*v)/(16.*(2 + np.sqrt(2))),
        (-2 + np.sqrt(2) + v - np.sqrt(2)*v)/(16.*(-2 + np.sqrt(2))),(-2 + np.sqrt(2) + (-1 + np.sqrt(2))*v)/(16.*(-2 + np.sqrt(2))),(2 + np.sqrt(2) + v + np.sqrt(2)*v)/(32 + 16*np.sqrt(2)),(-2 + np.sqrt(2) + v - np.sqrt(2)*v)/(16.*(-2 + np.sqrt(2))),
        (2 + np.sqrt(2) - (1 + np.sqrt(2))*v)/(16.*(2 + np.sqrt(2))),(2 + np.sqrt(2) - (1 + np.sqrt(2))*v)/(16.*(2 + np.sqrt(2))),(-2 + np.sqrt(2) + v - np.sqrt(2)*v)/(16.*(-2 + np.sqrt(2))),(2 + np.sqrt(2) + v + np.sqrt(2)*v)/(32 + 16*np.sqrt(2)),
        (-2 + np.sqrt(2) + (-1 + np.sqrt(2))*v)/(16.*(-2 + np.sqrt(2)))
        ])

    assert (np.abs(np.sum(p)-1.0) < (1E-6)),"Improperly normalized p!"
    return p


def get_werner(w):
    b1 = qt.bell_state('11')
    be = w*b1.proj()
    id = (1-w)/4*qt.tensor(qt.identity(2), qt.identity(2))
    return be + id

def getproj(qo,val):
    return 1/2*(val*qo + qt.tensor(qt.identity(1), qt.identity(1)))


def get_prob(a,b,x,y,w):
    aa = 1 if a == 0 else -1
    bb = 1 if b == 0 else -1
    xx = qt.sigmaz() if x == 0 else qt.sigmax()
    yy = 1/np.sqrt(2) * ((qt.sigmaz() + qt.sigmax()) if y == 0 else (qt.sigmaz() - qt.sigmax()))
    xx_p = getproj(xx,aa)
    yy_p = getproj(yy,bb)
    pro = qt.tensor(xx_p, yy_p)
    wer = get_werner(w)
    ##print(wer)
    ##print(pro)
    return (wer*pro).tr()/4

def target_distribution_gen2(name, parameter1, parameter2):
    """ parameter1 is usually a parameter of distribution (not always relevant). parameter2 is usually noise."""
    if name=="CHSH":
        v = parameter2
        arr = []
        for x in range(2):
            for y in range(2):
                for a in range(2):
                    for b in range(2):
                        arr.append(get_prob(a,b,x,y,v))
        p = np.array(arr)
    assert (np.abs(np.sum(p)-1.0) < (1E-6)),"Improperly normalized p!"
    return p



import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda
from scipy.stats import entropy
from tensorflow.keras.initializers import VarianceScaling



def build_model():
    """ Build NN """
    inputTensor = Input((3,))
    group_lambda = Lambda(lambda x: x[:,:1], output_shape=((1,)))(inputTensor)
    group_x_hidden = Lambda(lambda x: x[:,1:2], output_shape=((1,)))(inputTensor) # a input
    group_y_hidden = Lambda(lambda x: x[:,2:3], output_shape=((1,)))(inputTensor) # c input

    ais = 2
    bis = 2

    group_x_hidden = Lambda(lambda x:K.squeeze(K.one_hot(K.cast(x,'int32'), ais),axis=1) , output_shape=((ais,)))(group_x_hidden)
    group_y_hidden = Lambda(lambda x:K.squeeze(K.one_hot(K.cast(x,'int32'), bis),axis=1) , output_shape=((bis,)))(group_y_hidden)

    group_x = group_x_hidden
    group_y = group_y_hidden

    amean = ais/2
    astd = np.sqrt((ais**2-1)/12)
    bmean = bis/2
    bstd = np.sqrt((bis**2-1)/12)
    activ = 'relu' # activation for most of NN
    activ2 = 'softmax'
    weight_init_scaling = 2
    kernel_reg=None
    group_x_hidden = Lambda(lambda x: (x-amean)/astd , output_shape=((ais,)))(group_x_hidden)
    group_y_hidden = Lambda(lambda x: (x-bmean)/bstd , output_shape=((bis,)))(group_y_hidden)

    group_a1 = Concatenate()([group_lambda,group_x_hidden])
    group_b1 = Concatenate()([group_lambda,group_y_hidden])
    group_a2 = Concatenate()([group_lambda,group_x_hidden])
    group_b2 = Concatenate()([group_lambda,group_y_hidden])
    group_p = Concatenate()([group_lambda,group_x_hidden])

    ## Note: increasing the variance of the initialization seemed to help in some cases, especially when the number if outputs per party is 4 or more.
    kernel_init = VarianceScaling(scale=weight_init_scaling, mode='fan_in', distribution='truncated_normal', seed=None)

    for _ in range(3):
        group_a1 = Dense(8,activation=activ, kernel_regularizer=kernel_reg, kernel_initializer = kernel_init)(group_a1)
        group_b1 = Dense(8,activation=activ, kernel_regularizer=kernel_reg, kernel_initializer = kernel_init)(group_b1)
        group_a2 = Dense(8,activation=activ, kernel_regularizer=kernel_reg, kernel_initializer = kernel_init)(group_a2)
        group_b2 = Dense(8,activation=activ, kernel_regularizer=kernel_reg, kernel_initializer = kernel_init)(group_b2)
        group_p = Dense(8,activation=activ, kernel_regularizer=kernel_reg, kernel_initializer = kernel_init)(group_p)
        

    group_a1 = Dense(2,activation=activ2, kernel_regularizer=kernel_reg)(group_a1)
    group_b1 = Dense(2,activation=activ2, kernel_regularizer=kernel_reg)(group_b1)
    group_a2 = Dense(2,activation=activ2, kernel_regularizer=kernel_reg)(group_a2)
    group_b2 = Dense(2,activation=activ2, kernel_regularizer=kernel_reg)(group_b2)
    group_p = Dense(1,activation=activ2, kernel_regularizer=kernel_reg)(group_p)

    def doThis(layer1, layer2, layerp, stid):
        group_xx = Lambda(lambda x: x[:,stid:stid+1], output_shape=((1,)))(layer1)
        group_yy = Lambda(lambda x: x[:,stid:stid+1], output_shape=((1,)))(layer2)
        return Lambda(lambda x: x[0]*x[1] + (1-x[0])*x[2])([layerp, group_xx, group_yy])
    
    outputTensor = Concatenate()([group_x, group_y, doThis(group_a1, group_a2, group_p, 0),
    doThis(group_a1, group_a2, group_p, 1),doThis(group_b1, group_b2, group_p, 0),
    doThis(group_b1, group_b2, group_p, 1)])


   # def comfunc1(x) :
    #    return x[:][0]*x[4] + x[2]*(1-x[4])
    #def comfunc2(x):
     #   return x[1]*x[4]+x[3]*(1-x[4])

   # group_a11 = Lambda(comfunc1, output_shape= ((1,)))(group_aa)
   # group_b11 = Lambda(comfunc1, output_shape= ((1,)))(group_bb)
   # group_a22 = Lambda(comfunc2, output_shape= ((1,)))(group_aa)
   # group_b22 = Lambda(comfunc2, output_shape= ((1,)))(group_bb)

    #group_a = Concatenate()([group_a11, group_a22])
   # group_b = Concatenate()([group_b11, group_b22])
    #outputTensor = Concatenate()([group_x,group_y,group_a1,group_b1, group_a_2])

    model = Model(inputTensor,outputTensor)





    print(model.summary())
    return model
@tf.function
def f(x):
    arr = []
    for xs in x:
        arr.append([xs[0]*xs[1]])
    return arr

    
def doX():
    input = Input((3,))
    dd = Lambda(lambda x: x, output_shape = (3,))(input) 
    output = Lambda(f, output_shape = (1,))(dd) 
    model = Model(input, output)
    print(model.summary())
#build_model()