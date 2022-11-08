import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda
from scipy.stats import entropy
from tensorflow.keras.initializers import VarianceScaling
import qutip as qt
import random

batch_size = 50

def get_projector(vector, res):
    dot_pro = qt.sigmax()*vector[0] + qt.sigmay()*vector[1] + qt.sigmaz()*vector[2]
    return 1/2*(qt.identity(2) + dot_pro * (-1 if res == 0 else 1))

def compute_target_distribution(x_vec, y_vec):
    quantum_var = qt.bell_state('11').proj()
    return [
        (quantum_var*qt.tensor(get_projector(x_vec, a), get_projector(y_vec, b))).tr()
        for a in range(2) for b in range(2)
    ]

def build_model():
    input_tensor = Input((7,))
    group_lambda = Lambda(lambda x: x[:, :1], outputs_shape = ((1,)))(input_tensor)
    input_x = Lambda(lambda x: x[:, 1:4], outputs_shape = ((3,)))(input_tensor)
    input_y = Lambda(lambda x: x[:, 4:], outputs_shape = ((3,)))(input_tensor)
    group_a1 = Concatenate()([group_lambda, input_x])
    group_a2 = Concatenate()([group_lambda, input_x])
    group_p = Concatenate()([group_lambda, input_x])
    group_b1 = Concatenate()([group_lambda, input_y])
    group_b2 = Concatenate()([group_lambda, input_y])
    
    kernel_init = VarianceScaling(scale=2, mode='fan_in', distribution='truncated_normal', seed=None)
    act_func = "relu"
    width = 10
    regu = None
    for _ in range(5):
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

    output_tensor = Concatenate()([input_x, input_y, group_a1, group_a2, group_b1, group_b2, group_p])

    model = Model(input_tensor,output_tensor)
    return model

def generate_unit_vector():
    x = random.random()
    y = random.random()
    z = random.random()
    vec = [x,y,z]
    return vec/np.linalg.norm(vec)

def generate_xy_batch():
    while True:
        inputs = []
        answers = []
        for _ in range(batch_size):
            temp = [random.random()]
            vx = generate_unit_vector()
            vy = generate_unit_vector()
            input = np.concatenate((temp,vx,vy))
            answer = compute_target_distribution(vx,vy)
            inputs.append(input)
            answer.append(answer)
        yield (inputs, answer)

def generate_x_batch():
    while True:
        inputs = []
        for _ in range(batch_size):
            inputs.append(np.concatenate(([random.random()],generate_unit_vector(),generate_unit_vector())))
        yield inputs

def get_resultant_distr(a1,a2, b1, b2, p):
    return (np.outer(a1,b1)).flatten()*p + (1-p) * (np.outer(a2,b2))



