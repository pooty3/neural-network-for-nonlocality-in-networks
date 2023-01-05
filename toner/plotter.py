import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow.keras.backend as K
from tensorflow import keras 
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda
from scipy.stats import entropy
from tensorflow.keras.initializers import VarianceScaling
import qutip as qt
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
total_inputs = 300000


model_path = "my_model3.h5"
model = load_model(model_path, compile = False)
def normalis(x, y, z):
    vec = [x,y,z]
    return vec/np.linalg.norm(vec)
def generate_unit_vector(): 
    x = random.uniform(-1,1)
    y = random.uniform(-1,1)
    z = random.uniform(-1,1)
    vec = [x,y,z]
    return vec/np.linalg.norm(vec)

def get_polar(vec):
    [x,y,z] = vec
    phi = math.atan2(y, x)
    theta = math.acos(z)
    return [phi, theta]
def get_cart(phi, theta):
    z = math.cos(theta)
    x = math.sin(theta)*math.cos(phi)
    y = math.sin(theta)*math.sin(phi)
    return [x,y,z]

def generate_inputs(v1,v2):
    inputs = []
    for _ in range(total_inputs):
        alice = generate_unit_vector()
        bob = generate_unit_vector()
        input= np.concatenate((v1,v2, alice, bob))
        inputs.append(input.tolist())
    return inputs


def plotP(v1,v2, image_path):
    inputs = generate_inputs(v1,v2)
    results = model.predict(inputs)
    actual_results = []
    thetas = []
    phis = []
    val_p = []
    for idx in range(total_inputs):
        val = results[idx][8]
        alice_x = inputs[idx][6]
        alice_y = inputs[idx][7]
        alice_z = inputs[idx][8]
        phi = math.atan2(alice_y, alice_x)
        theta = math.acos(alice_z)
        thetas.append(theta)
        phis.append(phi)
        val_p.append(val)  
        actual_results.append((phi, theta, val))
    plt.scatter(x = phis, y = thetas, c = val_p,cmap = 'viridis')
    [p1, t1] = get_polar(v1)
    [p2, t2] = get_polar(v2)
    plt.plot([p1,p2], [t1,t2], marker =  "*", ls="none", ms=20)
    plt.colorbar()
    plt.savefig(image_path)
   # print(actual_results)


def plotvalue(v1, v2, value_idx, image_path):
    inputs = generate_inputs(v1,v2)
    results = model.predict(inputs)
    actual_results = []
    thetas = []
    phis = []
    val_p = []
    for idx in range(total_inputs):
        val = results[idx][value_idx]
        alice_x = inputs[idx][6]
        alice_y = inputs[idx][7]
        alice_z = inputs[idx][8]
        phi = math.atan2(alice_y, alice_x)
        theta = math.acos(alice_z)
        thetas.append(theta)
        phis.append(phi)
        val_p.append(val)  
        actual_results.append((phi, theta, val))
    plt.clf()
    plt.cla()
    plt.scatter(x = phis, y = thetas, c = val_p,cmap = 'viridis')
    [p1, t1] = get_polar(v1)
    [p2, t2] = get_polar(v2)
    plt.plot([p1,p2], [t1,t2], marker =  "*", ls="none", ms=20)
    plt.colorbar()
    plt.savefig(image_path)
#plotP([1,0,0], [0,1,0], "img44.png")


