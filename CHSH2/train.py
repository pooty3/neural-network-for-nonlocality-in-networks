import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import qr
import tensorflow.keras.backend as K
from tensorflow import keras 
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda
from scipy.stats import entropy
from tensorflow.keras.initializers import VarianceScaling
import qutip as qt
import random
import tensorflow as tf


from math import *
def get_werner2(w):
    b1 = qt.bell_state('11')
    be = w*b1.proj()
    id = (1-w)/4*qt.tensor(qt.identity(2), qt.identity(2))
    return be + id

def getproj2(qo,val):
    return 1/2*(val*qo + qt.tensor(qt.identity(1), qt.identity(1)))


def get_prob(a,b,x,y,w):
    aa = 1 if a == 0 else -1
    bb = 1 if b == 0 else -1
    xx = qt.sigmaz() if x == 0 else qt.sigmax()
    yy = 1/np.sqrt(2) * ((qt.sigmaz() + qt.sigmax()) if y == 0 else (qt.sigmaz() - qt.sigmax()))
    xx_p = getproj2(xx,aa)
    yy_p = getproj2(yy,bb)
    pro = qt.tensor(xx_p, yy_p)
    wer = get_werner2(w)
    ##print(wer)
    ##print(pro)
    return (wer*pro).tr()

def get_prob_now(x, y,w):
    arr = []
    for a in range(2):
        for b in range(2):
            arr.append(get_prob(a,b,x,y,w)) 
    return arr

def getCHSHprob(w):
    arr = []
    for x in range(2):
        for y in range(2):
            for a in range(2):
                for b in range(2):
                    arr.append(get_prob(a,b,x,y,w))
    return arr


hidden_variable_size = 5000

 
def build_model():
    input_tensor = Input((3,))
    group_lambda = Lambda(lambda x: x[:, :1], output_shape = ((1,)))(input_tensor)
    input_x = Lambda(lambda x: x[:, 1:2], output_shape = ((1,)))(input_tensor)
    input_y = Lambda(lambda x: x[:, 2:3], output_shape = ((1,)))(input_tensor)
    input_x = Lambda(lambda x:K.squeeze(K.one_hot(K.cast(x,'int32'), 2),axis=1) , output_shape=((2,)))(input_x)
    input_y = Lambda(lambda x:K.squeeze(K.one_hot(K.cast(x,'int32'), 2),axis=1) , output_shape=((2,)))(input_y)
    group_a = Concatenate()([group_lambda, input_x])
    group_b = Concatenate()([group_lambda, input_y])
    
    kernel_init = None
    act_func = "relu"
    width = 8
    regu = None
    for _ in range(3):
        group_a = Dense(width,activation=act_func, kernel_regularizer=regu, kernel_initializer = kernel_init)(group_a)
        group_b = Dense(width,activation=act_func, kernel_regularizer=regu, kernel_initializer = kernel_init)(group_b)

    
    group_a = Dense(2,activation="softmax", kernel_regularizer=None)(group_a)
    group_b = Dense(2,activation="softmax", kernel_regularizer=None)(group_b)


    output_tensor = Concatenate()([group_a, group_b])

    model = Model(input_tensor,output_tensor)
    return model


def build_model_communication():
    input_tensor = Input((3,))
    group_lambda = Lambda(lambda x: x[:, :1], output_shape = ((1,)))(input_tensor)
    input_x = Lambda(lambda x: x[:, 1:2], output_shape = ((1,)))(input_tensor)
    input_y = Lambda(lambda x: x[:, 2:3], output_shape = ((1,)))(input_tensor)
    input_x = Lambda(lambda x:K.squeeze(K.one_hot(K.cast(x,'int32'), 2),axis=1) , output_shape=((2,)))(input_x)
    input_y = Lambda(lambda x:K.squeeze(K.one_hot(K.cast(x,'int32'), 2),axis=1) , output_shape=((2,)))(input_y)
    group_a1 = Concatenate()([group_lambda, input_x])
    group_b1 = Concatenate()([group_lambda, input_y])
    group_a2 = Concatenate()([group_lambda, input_x])
    group_b2 = Concatenate()([group_lambda, input_y])
    group_p = Concatenate()([group_lambda, input_x])
    kernel_init = None
    act_func = "relu"
    width = 8
    regu = None
    for _ in range(3):
        group_a1 = Dense(width,activation=act_func, kernel_regularizer=regu, kernel_initializer = kernel_init)(group_a1)
        group_b1 = Dense(width,activation=act_func, kernel_regularizer=regu, kernel_initializer = kernel_init)(group_b1)
        group_a2 = Dense(width,activation=act_func, kernel_regularizer=regu, kernel_initializer = kernel_init)(group_a2)
        group_b2 = Dense(width,activation=act_func, kernel_regularizer=regu, kernel_initializer = kernel_init)(group_b2)
        group_p = Dense(width,activation=act_func, kernel_regularizer=regu, kernel_initializer = kernel_init)(group_p)
    
    group_a1 = Dense(2,activation="softmax", kernel_regularizer=None)(group_a1)
    group_b1 = Dense(2,activation="softmax", kernel_regularizer=None)(group_b1)
    group_a2 = Dense(2,activation="softmax", kernel_regularizer=None)(group_a2)
    group_b2 = Dense(2,activation="softmax", kernel_regularizer=None)(group_b2)
    group_p = Dense(1,activation="sigmoid", kernel_regularizer=None)(group_p)

    output_tensor = Concatenate()([group_a1, group_b1, group_a2, group_b2, group_p])

    model = Model(input_tensor,output_tensor)
    return model

def generate_xy_batch(w):
    while True:
        inputs = []
        answers = []
        for alice_idx in range(2):
            for bob_idx in range(2):
                answer = get_prob_now(alice_idx, bob_idx, w)
                for _ in range(hidden_variable_size):
                    temp = random.uniform(0,100)
                    input= [temp, alice_idx, bob_idx]
                    inputs.append(input)
                    answers.append(answer)
        res = (np.array(inputs), np.array(answers))
        yield res


def get_resultant_distr(a1,a2, b1, b2, p):
    return (np.outer(a1,b1)).flatten()*p + (1-p) * (np.outer(a2,b2))

def keras_distance(p,q):
    """ Distance used in loss function.
        kl: Kullback-Liebler divergence (relative entropy)
    """

    p = K.clip(p, K.epsilon(), 1)
    q = K.clip(q, K.epsilon(), 1)
    return K.sum(p * K.log(p / q), axis=-1)
    # p = K.clip(p, K.epsilon(), 1)
    # q = K.clip(q, K.epsilon(), 1)
    # avg = (p+q)/2
    # return K.sum(p * K.log(p / avg), axis=-1) + K.sum(q * K.log(q / avg), axis=-1)



def compute_loss(y_true, y_pred):
    loss = 0
    for idx in range(4):
        a = y_pred[idx*(hidden_variable_size) : (idx+1)*hidden_variable_size,0:2]
        b = y_pred[idx*(hidden_variable_size) : (idx+1)*hidden_variable_size,2:4]

        a_probs = K.reshape(a,(-1,2,1))
        b_probs = K.reshape(b,(-1,1,2))
        c_probs = a_probs*b_probs
        final = K.mean(c_probs,axis= 0)
        #print(final)
        final = K.flatten(final)
        #print(final)
        err = keras_distance(y_true[idx*(hidden_variable_size), :], final)
        loss += err
    return loss/4

def compute_loss_communication(y_true, y_pred):
    loss = 0
    for idx in range(4):
        a1 = y_pred[idx*(hidden_variable_size) : (idx+1)*hidden_variable_size,0:2]
        b1 = y_pred[idx*(hidden_variable_size) : (idx+1)*hidden_variable_size,2:4]
        a2 = y_pred[idx*(hidden_variable_size) : (idx+1)*hidden_variable_size,4:6]
        b2 = y_pred[idx*(hidden_variable_size) : (idx+1)*hidden_variable_size,6:8]
        p = y_pred[idx*(hidden_variable_size) : (idx+1)*hidden_variable_size,8:9]

        a1_probs = K.reshape(a1,(-1,2,1))
        b1_probs = K.reshape(b1,(-1,1,2))
        c1 = a1_probs*b1_probs
        a2_probs = K.reshape(a2,(-1,2,1))
        b2_probs = K.reshape(b2,(-1,1,2))
        c2 = a2_probs*b2_probs
        pp = K.reshape(p, (-1,1,1))
        final = pp*c1 + (1-pp)*c2
        final = K.mean(final,axis= 0)
        #print(final)
        final = K.flatten(final)
        err = keras_distance(y_true[idx*(hidden_variable_size), :], final)
        loss += err
    return loss/4

no_of_epochs = 100
batch_per_epochs = 100
v_steps = 25
space1 = [1.0, 0.7,0.75, 0.33, 0.0, 0.5, 0.6,0.65, 0.67, 0.68, 0.69]
space2 = [0.705, 0.706, 0.707, 0.708, 0.709, 0.71, 0.72, 0.73, 0.8, 0.9]

touse = space1

def run_model(img_path, data_path):
    entangled_amount = []
    results = []
    for w in touse:
        print(w)
        model = build_model()
    #model = build_model2()
        #keras.utils.plot_model(model, show_shapes=True)
        optimizer = "adam"
        model.compile(loss = compute_loss, optimizer = optimizer, metrics = [])
        model.fit(generate_xy_batch(w), steps_per_epoch = batch_per_epochs, epochs = no_of_epochs, verbose=1, validation_data=generate_xy_batch(w), validation_steps=v_steps, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=False, initial_epoch=0)
        result = model.evaluate(generate_xy_batch(w), steps = 100)
        entangled_amount.append(w)
        results.append(result)
    f = open(data_path, 'w+')
    f.write(str(entangled_amount))
    f.write(str(results))
    f.close()
    plt.clf()
    plt.scatter(entangled_amount, results)
    plt.savefig(img_path)

def run_model_communication(img_path, data_path):
    entangled_amount = []
    results = []
    for w in touse:
        print(w, "comm")
        model = build_model_communication()
    #model = build_model2()
        #keras.utils.plot_model(model, show_shapes=True)
        optimizer = "adam"
        model.compile(loss = compute_loss_communication, optimizer = optimizer, metrics = [])
        model.fit(generate_xy_batch(w), steps_per_epoch = batch_per_epochs, epochs = no_of_epochs, verbose=1, validation_data=generate_xy_batch(w), validation_steps=v_steps, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=False, initial_epoch=0)
        result = model.evaluate(generate_xy_batch(w), steps = 100)
        entangled_amount.append(w)
        results.append(result)
    f = open(data_path, 'w+')
    f.write(str(entangled_amount))
    f.write(str(results))
    f.close()
    plt.clf()
    plt.scatter(entangled_amount, results)
    plt.savefig(img_path)


run_model_communication("chsh2_comm.png", "chsh_comm.txt")
#run_model("chsh22.png", "chsh22.txt")