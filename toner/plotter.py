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

def normalis(x, y, z):
    vec = [x,y,z]
    return vec/np.linalg.norm(vec)
def generate_unit_vector(): 
    x = random.uniform(-1,1)
    y = random.uniform(-1,1)
    z = random.uniform(-1,1)
    vec = [x,y,z]
    return vec/np.linalg.norm(vec)

def generate_inputs():
    inputs = []
    for _ in range(total_inputs):
        alice = generate_unit_vector()
        bob = generate_unit_vector()
        v1 = normalis(0.5,0.5,0.5)
        v2 = normalis(1.0,-1.0,0.5)
        input= np.concatenate((v1,v2, alice, bob))
        inputs.append(input.tolist())
    return inputs


def plot():
    model = load_model('my_model', compile = False)
    inputs = generate_inputs()

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
    plt.colorbar()
    plt.savefig("my_img2")
    print(actual_results)


plot()