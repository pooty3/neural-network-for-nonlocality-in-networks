import numpy as np
import matplotlib.pyplot as plt

import tensorflow.keras.backend as K
from tensorflow import keras 
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda
from scipy.stats import entropy
from tensorflow.keras.initializers import VarianceScaling
import qutip as qt
import random
import tensorflow as tf
batch_size = 80
hidden_var_size = 500
def get_projector(vector, res):
    dot_pro = qt.sigmax()*vector[0] + qt.sigmay()*vector[1] + qt.sigmaz()*vector[2]
    return 1/2*(qt.identity(2) + dot_pro * (-1 if res == 0 else 1))

def compute_target_distribution(x_vec, y_vec):
    quantum_var = qt.bell_state('11').proj()
    return [
        (quantum_var*qt.tensor(get_projector(x_vec, a), get_projector(y_vec, b))).tr()
        for a in range(2) for b in range(2)
    ]

def build_model2():
    input_tensor = Input((12,))
    output_tensor = input_tensor
    for _ in range(10):
        output_tensor = Dense(10, activation="relu")(output_tensor)
    group_a1 = Dense(2,activation="softmax", kernel_regularizer=None)(output_tensor)
    group_a2 = Dense(2,activation="softmax", kernel_regularizer=None)(output_tensor)
    group_b1 = Dense(2,activation="softmax", kernel_regularizer=None)(output_tensor)
    group_b2 = Dense(2,activation="softmax", kernel_regularizer=None)(output_tensor)
    p = Dense(1, activation="sigmoid")(output_tensor)
    output_tensor1 = Concatenate()([group_a1, group_a2, group_b1, group_b2, p])
    model = Model(input_tensor,output_tensor1)
    return model



def build_model():
    input_tensor = Input((12,))
    group_lambda = Lambda(lambda x: x[:, :6], output_shape = ((6,)))(input_tensor)
    input_x = Lambda(lambda x: x[:, 6:9], output_shape = ((3,)))(input_tensor)
    input_y = Lambda(lambda x: x[:, 9:12], output_shape = ((3,)))(input_tensor)
    group_a1 = Concatenate()([group_lambda, input_x])
    group_a2 = Concatenate()([group_lambda, input_x])
    group_p = Concatenate()([group_lambda, input_x])
    group_b1 = Concatenate()([group_lambda, input_y])
    group_b2 = Concatenate()([group_lambda, input_y])
    
    kernel_init = None
    act_func = "relu"
    width = 30
    regu = None
    for _ in range(3):
        group_a1 = Dense(width,activation=act_func, kernel_regularizer=regu, kernel_initializer = kernel_init)(group_a1)
        group_a2 = Dense(width,activation=act_func, kernel_regularizer=regu, kernel_initializer = kernel_init)(group_a2)
        group_b1 = Dense(width,activation=act_func, kernel_regularizer=regu, kernel_initializer = kernel_init)(group_b1)
        group_b2 = Dense(width,activation=act_func, kernel_regularizer=regu, kernel_initializer = kernel_init)(group_b2)
        group_p = Dense(width,activation=act_func, kernel_regularizer=regu, kernel_initializer = kernel_init)(group_p)
    
    group_a1 = Dense(2,activation="softmax", kernel_regularizer=None)(group_a1)
    group_a2 = Dense(2,activation="softmax", kernel_regularizer=None)(group_a2)
    group_b1 = Dense(2,activation="softmax", kernel_regularizer=None)(group_b1)
    group_b2 = Dense(2,activation="softmax", kernel_regularizer=None)(group_b2)
    group_p = Dense(1,activation="sigmoid", kernel_regularizer=None)(group_p)

    output_tensor = Concatenate()([group_a1, group_a2, group_b1, group_b2, group_p])

    model = Model(input_tensor,output_tensor)
    return model

def generate_unit_vector(): 
    x = random.uniform(-1,1)
    y = random.uniform(-1,1)
    z = random.uniform(-1,1)
    vec = [x,y,z]
    return vec/np.linalg.norm(vec)

def generate_xy_batch():
    while True:
        inputs = []
        answers = []
        for _ in range(batch_size):
            vx = generate_unit_vector()
            vy = generate_unit_vector()
            answer = compute_target_distribution(vx,vy)
            for _ in range(hidden_var_size):
                temp1 = generate_unit_vector()
                temp2 = generate_unit_vector()
                input= np.concatenate((temp1,temp2, vx,vy))
                inputs.append(input.tolist())
                answers.append(answer)
        res = (np.array(inputs), np.array(answers))
        yield res

def generate_x_batch():
    while True:
        inputs = []
        for _ in range(batch_size):
            inputs.append(np.concatenate(([random.random()],generate_unit_vector(),generate_unit_vector())))
        yield inputs

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
    for idx in range(batch_size):
        a1 = y_pred[idx*(hidden_var_size) : (idx+1)*hidden_var_size,0:2]
        a2 = y_pred[idx*(hidden_var_size) : (idx+1)*hidden_var_size,2:4]
        b1 = y_pred[idx*(hidden_var_size) : (idx+1)*hidden_var_size,4:6]
        b2 = y_pred[idx*(hidden_var_size) : (idx+1)*hidden_var_size,6:8]
        p = y_pred[idx*(hidden_var_size) : (idx+1)*hidden_var_size,8:9]
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
        #print(final)
        err = keras_distance(final, y_true[idx*(hidden_var_size), :])
        loss += err
    return loss/batch_size


def build_and_save_model(file_path):
    model = build_model()
    #model = build_model2()
    keras.utils.plot_model(model, show_shapes=True)
    optimizer = "adam"
    model.compile(loss = compute_loss, optimizer = optimizer, metrics = [])
    model.fit_generator(generate_xy_batch(), steps_per_epoch=100, epochs=100, verbose=1, validation_data=generate_xy_batch(), validation_steps=20, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=False, initial_epoch=0)
    model.save("my_model")

build_and_save_model("my_model")

