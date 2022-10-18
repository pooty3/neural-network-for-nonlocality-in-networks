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

