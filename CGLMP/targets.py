from random import getstate
import numpy as np
from itertools import product

from scipy.io import loadmat

import qutip as qt
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
'''
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
'''

def getentangledstate():
    gamma = (np.sqrt(11) - np.sqrt(3))/2
    s1 = qt.basis(3,0)
    s2 = qt.basis(3,1)
    s3 = qt.basis(3,2)
    st = (qt.tensor(s1,s1) + qt.tensor(s2,s2) + gamma*qt.tensor(s3,s3)).unit()
    return st
def get_werner(w):
    entangled_part = w*getentangledstate().proj()
    noise = qt.tensor(qt.identity(3), qt.identity(3))/9 * (1-w)
    return noise + entangled_part
    #b1 = qt.bell_state('11')
    #be = w*b1.proj()
    ##id = (1-w)/4*qt.identity(4)
    #ff = be.data + id.data
    #return qt.Qobj(ff)


def getaliceproj(input, output):
    m = 3
    alpha_x = 0 if input == 0 else np.pi/m
    expo = 1j*alpha_x + 1j*2*np.pi*output/m
    totalstate = 1/np.sqrt(m) *(qt.basis(3,0)*np.exp(expo*0) + qt.basis(3,1)*np.exp(expo*1) + qt.basis(3,2)*np.exp(expo*2))
    return totalstate.proj()

def getbobproj(input, output):
    m = 3
    beta_y = -np.pi/(2*m) if input == 0 else np.pi/(2*m)
    expo = 1j*beta_y + 1j*2*np.pi*output/m
    totalstate = 1/np.sqrt(m) *(qt.basis(3,0)*np.exp(expo*0) + qt.basis(3,1)*np.exp(expo*1) + qt.basis(3,2)*np.exp(expo*2))
    return totalstate.proj()

def get_prob(a,b,x,y,w):

    wer = get_werner(w)

    measure = qt.tensor(getaliceproj(x,a), getbobproj(y,b))
    return (wer*measure).tr()/4

def target_distribution_gen(name, parameter1, parameter2):
    """ parameter1 is usually a parameter of distribution (not always relevant). parameter2 is usually noise."""
    if name=="CGLMP":
        v = parameter2
        arr = []
        for x in range(2):
            for y in range(2):
                for a in range(3):
                    for b in range(3):
                        arr.append(get_prob(a,b,x,y,v))
        p = np.real(np.array(arr))
    assert (np.abs(np.sum(p)-1.0) < (1E-6)),"Improperly normalized p!"
    return p

print(target_distribution_gen("CGLMP", 1, 0.5))