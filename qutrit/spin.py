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
from math import *
hidden_variable_size = 500
batch_size = 200

 
def generate_unit_vector(): 
    x = random.uniform(-1,1)
    y = random.uniform(-1,1)
    z = random.uniform(-1,1)
    vec = [x,y,z]
    return vec/np.linalg.norm(vec)

def convert_to_spherical(vec):
    return [acos(vec[2]), atan2(vec[1], vec[0])]


def build_model():
    input_tensor = Input((8,))
    group_lambda = Lambda(lambda x: x[:, :4], output_shape = ((4,)))(input_tensor)
    input_x = Lambda(lambda x: x[:, 4:6], output_shape = ((2,)))(input_tensor)
    input_y = Lambda(lambda x: x[:, 6:8], output_shape = ((2,)))(input_tensor)
    group_a = Concatenate()([group_lambda, input_x])
    group_b = Concatenate()([group_lambda, input_y])
    
    kernel_init = None
    act_func = "relu"
    width = 20
    regu = None
    for _ in range(3):
        group_a = Dense(width,activation=act_func, kernel_regularizer=regu, kernel_initializer = kernel_init)(group_a)
        group_b = Dense(width,activation=act_func, kernel_regularizer=regu, kernel_initializer = kernel_init)(group_b)

    
    group_a = Dense(3,activation="softmax", kernel_regularizer=None)(group_a)
    group_b = Dense(3,activation="softmax", kernel_regularizer=None)(group_b)


    output_tensor = Concatenate()([group_a, group_b])

    model = Model(input_tensor,output_tensor)
    return model


def build_model_communication():
    input_tensor = Input((8))
    group_lambda = Lambda(lambda x: x[:, :4], output_shape = ((4,)))(input_tensor)
    input_x = Lambda(lambda x: x[:, 4:6], output_shape = ((2,)))(input_tensor)
    input_y = Lambda(lambda x: x[:, 6:8], output_shape = ((2,)))(input_tensor)
    group_a1 = Concatenate()([group_lambda, input_x])
    group_b1 = Concatenate()([group_lambda, input_y])
    group_a2 = Concatenate()([group_lambda, input_x])
    group_b2 = Concatenate()([group_lambda, input_y])
    group_p = Concatenate()([group_lambda, input_x])
    kernel_init = None
    act_func = "relu"
    width = 20
    regu = None
    for _ in range(3):
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
        for _ in range(batch_size):
            vx = generate_unit_vector()
            vy = generate_unit_vector()
            sx = convert_to_spherical(vx)
            sy = convert_to_spherical(vy)
            answer = db.get_spin_probs(vx,vy,w, N = 3)
            for _ in range(hidden_variable_size):
                temp1 = convert_to_spherical(generate_unit_vector())
                temp2 = convert_to_spherical(generate_unit_vector())
                #input = np.concatenate(([random.uniform(0,1)], sx, sy))
                input= np.concatenate((temp1,temp2, sx, sy))
                inputs.append(input.tolist())
                answers.append(answer)
        res = (np.array(inputs), np.array(answers))
        yield res

# def generate_x_batch():
#     while True:
#         inputs = []
#         for _ in range(batch_size):
#             inputs.append(np.concatenate(([random.random()],generate_unit_vector(),generate_unit_vector())))
#         yield inputs

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
    for idx in range(batch_size):
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
    return loss/(batch_size)

def compute_loss_communication(y_true, y_pred):
    loss = 0
    for idx in range(batch_size):
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
    return loss/(batch_size)

no_of_epochs = 100
batch_per_epochs = 30
data_points = 13

wspace = [0, 0.3, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
def run_model(img_path, data_path):
    entangled_amount = []
    results = []
    for w in wspace:
        print(w)
        model = build_model()
    #model = build_model2()
        #keras.utils.plot_model(model, show_shapes=True)
        optimizer = "adam"
        model.compile(loss = compute_loss, optimizer = optimizer, metrics = [])
        model.fit(generate_xy_batch(w), steps_per_epoch = batch_per_epochs, epochs = no_of_epochs, verbose=1, validation_data=generate_xy_batch(w), validation_steps=5, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=False, initial_epoch=0)
        result = model.evaluate(generate_xy_batch(w), steps = 40)
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
    for w in wspace:
        print(w, "comm")
        model = build_model_communication()
    #model = build_model2()
        #keras.utils.plot_model(model, show_shapes=True)
        optimizer = "adam"
        model.compile(loss = compute_loss_communication, optimizer = optimizer, metrics = [])
        model.fit(generate_xy_batch(w), steps_per_epoch = batch_per_epochs, epochs = no_of_epochs, verbose=1, validation_data=generate_xy_batch(w), validation_steps=5, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=False, initial_epoch=0)
        result = model.evaluate(generate_xy_batch(w), steps = 40)
        entangled_amount.append(w)
        results.append(result)
    f = open(data_path, 'w+')
    f.write(str(entangled_amount))
    f.write(str(results))
    f.close()
    plt.clf()
    plt.scatter(entangled_amount, results)
    plt.savefig(img_path)


def train_and_save_comm(model_path):
    model = build_model_communication()
    #model = build_model2()
    optimizer = "adam"
    model.compile(loss = compute_loss, optimizer = optimizer, metrics = [])
    model.fit(generate_xy_batch(1.0), steps_per_epoch = batch_per_epochs, epochs = 100, verbose=1, validation_data=generate_xy_batch(1.0), validation_steps=5, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=False, initial_epoch=0)
    model.save(model_path)

def train_and_save_non_comm(model_path):
    model = build_model()
    #model = build_model2()
    optimizer = "adam"
    model.compile(loss = compute_loss, optimizer = optimizer, metrics = [])
    model.fit(generate_xy_batch(1.0), steps_per_epoch = batch_per_epochs, epochs = 25, verbose=1, validation_data=generate_xy_batch(1.0), validation_steps=5, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=False, initial_epoch=0)
    gg = model.evaluate(generate_xy_batch(1.0), steps = 100)
    f = open("some_data.txt", 'w+')
    f.write(gg)
    f.close()
    model.save(model_path)

def generate_unit_vector(): 
    x = random.uniform(-1,1)
    y = random.uniform(-1,1)
    z = random.uniform(-1,1)
    vec = [x,y,z]
    return vec/np.linalg.norm(vec)

def generate_inputs(v1,v2, total_inputs):
    inputs = []
    vv1 = convert_to_spherical(v1)
    vv2 = convert_to_spherical(v2)
    for _ in range(total_inputs):
        alice = convert_to_spherical(generate_unit_vector())
        bob = convert_to_spherical(generate_unit_vector())
        input= np.concatenate((vv1,vv2,alice,bob))
        inputs.append(input.tolist())
    return inputs

def get_data():
    v1 = [0,0,1]
    v2 = [0,1,0]
    inputs = generate_inputs(v1,v2, 100000)
    model = load_model("spin_model_kl.h5", compile = False)
    results = model.predict(inputs)
    f = open("some_data.txt", 'w')
    for idx in range(len(results)):
        for da in inputs[idx]:
            f.write(f'{da:.5f} ')
        for da in results[idx]:
            f.write(f'{da: .5f} ')
        f.write('\n')

def test():
    model = load_model("spin_model_kl.h5", compile = False)
    model.compile(loss = compute_loss, optimizer = "adam", metrics = [])
    result = model.evaluate(generate_xy_batch(1.0), steps = 100)
    print(result)
#train_and_save_non_comm("spin_model_non_kl.h5")

#get_data()
test()