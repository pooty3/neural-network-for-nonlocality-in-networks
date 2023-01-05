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
import distributions as db

hidden_variable_size = 1000
no_of_settings = 2
'''
alice_unitaries = [
    qt.Qobj([[ 0.61551508+0.36591687j, -0.65603294+0.04648796j, -0.22843137-0.05024736j],
 [ 0.01028608+0.36349963j,  0.10782052+0.01521212j,  0.10115754+0.91960464j],
 [-0.30709949+0.51058609j,  0.00141703+0.74538646j,  0.18898702-0.23167364j]]),
    qt.Qobj([[-0.41260565-0.05201496j,  0.55201172+0.68778569j, -0.14168517+0.17091004j],
 [ 0.40494024+0.197483j,  -0.14353321+0.20069598j,  0.17619665+0.83970111j],
 [ 0.78994511-0.00784872j,  0.35881464+0.1805694j,  -0.36581826-0.28416344j]]),
qt.Qobj(
 [[-0.65705736+0.1042858j,   0.50328138-0.15232931j, -0.29593718+0.43968729j],
 [-0.68893152-0.02108102j, -0.17246709+0.13535825j,  0.56105687-0.40258836j],
 [ 0.04388207-0.28355495j,  0.29265845-0.76797651j, -0.03385351-0.49100714j]]),
 qt.Qobj(
 [[ 0.42320766+0.56422224j, -0.42649653-0.42000041j, -0.05688164+0.37551748j],
 [ 0.18278271+0.23197051j, -0.19986778-0.13391533j,  0.05515338-0.92296144j],
 [-0.34302916+0.54558198j, -0.28926258+0.70721077j, -0.00088715+0.02916493j]])
]
bob_unitaries = [
    qt.Qobj([[-0.64029055-0.39441247j,  0.02867879+0.5714175j,  -0.25562342-0.20440904j],
 [-0.58840946-0.11496561j, -0.44166414-0.56454095j,  0.05892321+0.35115748j],
 [ 0.24751357+0.1173076j,  -0.38176632+0.11463137j, -0.82386951+0.295515j  ]]),

 qt.Qobj([[-0.18250912-0.18280534j,  0.28248861-0.25560301j, -0.06296543-0.88553671j],
 [ 0.1262268 -0.47276623j,  0.74221284+0.02416311j, -0.32237771+0.32429577j],
 [-0.61967083+0.55663235j,  0.43196432+0.34177531j,  0.01256369+0.05105981j]]),

qt.Qobj(
 [[-0.64176169+0.154509j,   -0.21830749+0.63215857j, -0.33623449-0.06271094j],
 [-0.19671543-0.49027801j,  0.36888779+0.39848806j,  0.6243613 +0.19034848j],
 [-0.11426662-0.52167285j,  0.48086477-0.16314262j, -0.66410018+0.12619463j]]),
qt.Qobj(
 [[ 0.30092735-0.26071382j, -0.88189906+0.22481545j, -0.07276904-0.08881305j],
 [ 0.53159216+0.44306896j,  0.04278361+0.30318679j,  0.44754531+0.47649785j],
 [ 0.16734874-0.57841603j,  0.10903907-0.2570406j,  -0.22269029+0.71405808j]])
]
'''
alice_unitaries = db.getaliceGCLMP()
bob_unitaries = db.getBobGCLMP()
def build_model():
    input_tensor = Input((3,))
    group_lambda = Lambda(lambda x: x[:, :1], output_shape = ((1,)))(input_tensor)
    input_x = Lambda(lambda x: x[:, 1:2], output_shape = ((1,)))(input_tensor)
    input_y = Lambda(lambda x: x[:, 2:3], output_shape = ((1,)))(input_tensor)
    input_x = Lambda(lambda x:K.squeeze(K.one_hot(K.cast(x,'int32'), no_of_settings),axis=1) , output_shape=((no_of_settings,)))(input_x)
    input_y = Lambda(lambda x:K.squeeze(K.one_hot(K.cast(x,'int32'), no_of_settings),axis=1) , output_shape=((no_of_settings,)))(input_y)
    group_a = Concatenate()([group_lambda, input_x])
    group_b = Concatenate()([group_lambda, input_y])
    
    kernel_init = None
    act_func = "relu"
    width = 100
    regu = None
    for _ in range(5):
        group_a = Dense(width,activation=act_func, kernel_regularizer=regu, kernel_initializer = kernel_init)(group_a)
        group_b = Dense(width,activation=act_func, kernel_regularizer=regu, kernel_initializer = kernel_init)(group_b)

    
    group_a = Dense(3,activation="softmax", kernel_regularizer=None)(group_a)
    group_b = Dense(3,activation="softmax", kernel_regularizer=None)(group_b)


    output_tensor = Concatenate()([group_a, group_b])

    model = Model(input_tensor,output_tensor)
    return model


def build_model_communication():
    input_tensor = Input((3,))
    group_lambda = Lambda(lambda x: x[:, :1], output_shape = ((1,)))(input_tensor)
    input_x = Lambda(lambda x: x[:, 1:2], output_shape = ((1,)))(input_tensor)
    input_y = Lambda(lambda x: x[:, 2:3], output_shape = ((1,)))(input_tensor)
    input_x = Lambda(lambda x:K.squeeze(K.one_hot(K.cast(x,'int32'), no_of_settings),axis=1) , output_shape=((no_of_settings,)))(input_x)
    input_y = Lambda(lambda x:K.squeeze(K.one_hot(K.cast(x,'int32'), no_of_settings),axis=1) , output_shape=((no_of_settings,)))(input_y)
    group_a1 = Concatenate()([group_lambda, input_x])
    group_b1 = Concatenate()([group_lambda, input_y])
    group_a2 = Concatenate()([group_lambda, input_x])
    group_b2 = Concatenate()([group_lambda, input_y])
    group_p = Concatenate()([group_lambda, input_x])
    kernel_init = None
    act_func = "relu"
    width = 100
    regu = None
    for _ in range(5):
        group_a1 = Dense(width,activation=act_func, kernel_regularizer=regu, kernel_initializer = kernel_init)(group_a1)
        group_b1 = Dense(width,activation=act_func, kernel_regularizer=regu, kernel_initializer = kernel_init)(group_b1)
        group_a2 = Dense(width,activation=act_func, kernel_regularizer=regu, kernel_initializer = kernel_init)(group_a2)
        group_b2 = Dense(width,activation=act_func, kernel_regularizer=regu, kernel_initializer = kernel_init)(group_b2)
        group_p = Dense(width,activation=act_func, kernel_regularizer=regu, kernel_initializer = kernel_init)(group_p)
    
    group_a1 = Dense(3,activation="softmax", kernel_regularizer=None)(group_a1)
    group_b1 = Dense(3,activation="softmax", kernel_regularizer=None)(group_b1)
    group_a2 = Dense(3,activation="softmax", kernel_regularizer=None)(group_a2)
    group_b2 = Dense(3,activation="softmax", kernel_regularizer=None)(group_b2)
    group_p = Dense(1,activation="sigmoid", kernel_regularizer=None)(group_p)

    output_tensor = Concatenate()([group_a1, group_b1, group_a2, group_b2, group_p])

    model = Model(input_tensor,output_tensor)
    return model

def generate_xy_batch(w):
    while True:
        inputs = []
        answers = []
        for alice_idx in range(no_of_settings):
            for bob_idx in range(no_of_settings):
                alice_unitary = alice_unitaries[alice_idx]
                bob_unitary = bob_unitaries[bob_idx]
                answer = db.getprobabilities(alice_unitary, bob_unitary, w)
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
    avg = (p+q)/2
    return K.sum(p * K.log(p / avg), axis=-1) + K.sum(q * K.log(q / avg), axis=-1)



def compute_loss(y_true, y_pred):
    loss = 0
    for idx in range(no_of_settings*no_of_settings):
        a = y_pred[idx*(hidden_variable_size) : (idx+1)*hidden_variable_size,0:3]
        b = y_pred[idx*(hidden_variable_size) : (idx+1)*hidden_variable_size,3:6]

        a_probs = K.reshape(a,(-1,3,1))
        b_probs = K.reshape(b,(-1,1,3))
        c_probs = a_probs*b_probs
        final = K.mean(c_probs,axis= 0)
        #print(final)
        final = K.flatten(final)
        #print(final)
        err = keras_distance(y_true[idx*(hidden_variable_size), :], final)
        loss += err
    return loss/(no_of_settings*no_of_settings)

def compute_loss_communication(y_true, y_pred):
    loss = 0
    for idx in range(no_of_settings*no_of_settings):
        a1 = y_pred[idx*(hidden_variable_size) : (idx+1)*hidden_variable_size,0:3]
        b1 = y_pred[idx*(hidden_variable_size) : (idx+1)*hidden_variable_size,3:6]
        a2 = y_pred[idx*(hidden_variable_size) : (idx+1)*hidden_variable_size,6:9]
        b2 = y_pred[idx*(hidden_variable_size) : (idx+1)*hidden_variable_size,9:12]
        p = y_pred[idx*(hidden_variable_size) : (idx+1)*hidden_variable_size,12:13]

        a1_probs = K.reshape(a1,(-1,3,1))
        b1_probs = K.reshape(b1,(-1,1,3))
        c1 = a1_probs*b1_probs
        a2_probs = K.reshape(a2,(-1,3,1))
        b2_probs = K.reshape(b2,(-1,1,3))
        c2 = a2_probs*b2_probs
        pp = K.reshape(p, (-1,1,1))
        final = pp*c1 + (1-pp)*c2
        final = K.mean(final,axis= 0)
        #print(final)
        final = K.flatten(final)
        err = keras_distance(y_true[idx*(hidden_variable_size), :], final)
        loss += err
    return loss/(no_of_settings*no_of_settings)

def run_model(img_path):
    entangled_amount = []
    results = []
    for w in np.linspace(0,1,16, endpoint= True):
        print(w)
        model = build_model()
    #model = build_model2()
        #keras.utils.plot_model(model, show_shapes=True)
        optimizer = "adam"
        model.compile(loss = compute_loss, optimizer = optimizer, metrics = [])
        model.fit(generate_xy_batch(w), steps_per_epoch=50, epochs=200, verbose=1, validation_data=generate_xy_batch(w), validation_steps=5, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=False, initial_epoch=0)
        result = model.evaluate(generate_xy_batch(w), steps = 10)
        entangled_amount.append(w)
        results.append(result)
    plt.clf()
    plt.scatter(entangled_amount, results)
    plt.savefig(img_path)

def run_model_communication(img_path):
    entangled_amount = []
    results = []
    for w in np.linspace(0,1,16, endpoint= True):
        print(w, "comm")
        model = build_model_communication()
    #model = build_model2()
        #keras.utils.plot_model(model, show_shapes=True)
        optimizer = "adam"
        model.compile(loss = compute_loss_communication, optimizer = optimizer, metrics = [])
        model.fit(generate_xy_batch(w), steps_per_epoch=50, epochs=200, verbose=1, validation_data=generate_xy_batch(w), validation_steps=5, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=False, initial_epoch=0)
        result = model.evaluate(generate_xy_batch(w), steps = 10)
        entangled_amount.append(w)
        results.append(result)
    plt.clf()
    plt.scatter(entangled_amount, results)
    plt.savefig(img_path)

run_model_communication("loss_plot_comm.png")
#build_and_save_model("my_model3.h5")

