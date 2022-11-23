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
batch_size = 1
hidden_var_size = 8
def get_projector(vector, res):
    dot_pro = qt.sigmax()*vector[0] + qt.sigmay()*vector[1] + qt.sigmaz()*vector[2]
    return 1/2*(qt.identity(2) + dot_pro * (-1 if res == 0 else 1))

def compute_target_distribution(x_vec, y_vec):
    quantum_var = qt.bell_state('11').proj()
    return [
        (quantum_var*qt.tensor(get_projector(x_vec, a), get_projector(y_vec, b))).tr()
        for a in range(2) for b in range(2)
    ]

def printer(a):
    print("here!printer!")
    return a

def build_model():
    input_tensor1 = Input((7,))
    input_tensor = Lambda(lambda x: printer(x), output_shape = ((7,)))(input_tensor1)
    group_lambda = Lambda(lambda x: x[:, :1], output_shape = ((1,)))(input_tensor)
    input_x = Lambda(lambda x: x[:, 1:4], output_shape = ((3,)))(input_tensor)
    input_y = Lambda(lambda x: x[:, 4:], output_shape = ((3,)))(input_tensor)
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
            vx = generate_unit_vector()
            vy = generate_unit_vector()
            answer = compute_target_distribution(vx,vy)
            for _ in range(hidden_var_size):
                temp = [random.random()]
                input= np.concatenate((temp, vx,vy))
                inputs.append(input.tolist())
                answers.append(answer)
        print("giving batch...")
        print(inputs)
        res = (np.array(inputs), np.array(answers))
        print(res)
        yield res

def generate_x_batch():
    while True:
        inputs = []
        for _ in range(batch_size):
            inputs.append(np.concatenate(([random.random()],generate_unit_vector(),generate_unit_vector())))
        yield inputs

def get_resultant_distr(a1,a2, b1, b2, p):
    return (np.outer(a1,b1)).flatten()*p + (1-p) * (np.outer(a2,b2))

def compute_loss(y_true, y_pred):
    print("inside loss!")
    print(y_true)
    d = tf.Print(y_pred, [y_pred], "Inside loss function")
    return 1
    '''
    current_idx = 0
    total_loss = 0
    for _ in range(batch_size):
        current_distr = np.array([0,0,0,0])
        for _ in range(hidden_var_size):
            i1,i2, a1,a2,b1,b2,p = y_pred[current_idx]
            current_idx += 1
            distr = get_resultant_distr(a2,a2,b1,b2,p)
            current_distr += distr
        current_distr /= hidden_var_size
        total_loss += keras_distance(y_true[current_idx - 1], current_distr)
    return total_loss/batch_size
'''

def keras_distance(p,q):
    """ Distance used in loss function.
        kl: Kullback-Liebler divergence (relative entropy)
    """
    print(p)
    print(q)
    print("distance!")
    p = K.clip(p, K.epsilon(), 1)
    q = K.clip(q, K.epsilon(), 1)
    avg = (p+q)/2
    return K.sum(p * K.log(p / avg), axis=-1) + K.sum(q * K.log(q / avg), axis=-1)


def build_and_save_model(file_path):
    model = build_model()
    print(model.summary())
    keras.utils.plot_model(model, show_shapes=True)
    optimizer = "adadelta"
    model.compile(loss = compute_loss, optimizer = optimizer, metrics = [])
    model.fit_generator(generate_xy_batch(), steps_per_epoch=5, epochs=1, verbose=1, validation_data=generate_xy_batch(), validation_steps=5, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=False, initial_epoch=0)
    model.save("my_model")

build_and_save_model("my_model")