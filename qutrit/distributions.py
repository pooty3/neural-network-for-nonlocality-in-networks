import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import qr

import qutip as qt
import random

N = 3

def get_random_unitary2():
    return qt.rand_unitary_haar(3)

def projectors(U):
    new_proj = [(U*qt.basis(N, i)).proj() for i in range(N)]
    return new_proj
def get_maximally_entangled_state():
    return (1/np.sqrt(3)*sum([qt.tensor(qt.basis(N,i), qt.basis(N,i)) for i in range(N)])).proj()

def werner_state(w):
    s1 = get_maximally_entangled_state()*w
    s2 = (1-w)/9*qt.tensor(qt.identity(3), qt.identity(3))
    return s1+s2

def getprobabilities(alice_U, bob_U, w):
    quantum_resource = werner_state(w)
    alice_proj = projectors(alice_U)
    bob_proj = projectors(bob_U)
    probs = [np.abs((qt.tensor(alice_proj[i], bob_proj[j])*quantum_resource).tr()) for i in range(N) for j in range(N)]
    assert(np.abs(sum(probs) -1 ) < 0.00001)
    return probs

    


def getaliceproj(input, output):
    m = 3
    alpha_x = 0 if input == 0 else np.pi/m
    expo = 1j*alpha_x + 1j*2*np.pi*output/m
    totalstate = 1/np.sqrt(m) *(qt.basis(3,0)*np.exp(expo*0) + qt.basis(3,1)*np.exp(expo*1) + qt.basis(3,2)*np.exp(expo*2))
    return totalstate

def getbobproj(input, output):
    m = 3
    beta_y = -np.pi/(2*m) if input == 0 else np.pi/(2*m)
    expo = 1j*beta_y + 1j*2*np.pi*output/m
    totalstate = 1/np.sqrt(m) *(qt.basis(3,0)*np.exp(expo*0) + qt.basis(3,1)*np.exp(expo*1) + qt.basis(3,2)*np.exp(expo*2))
    return totalstate

def getaliceGCLMP():
    return [
        qt.Qobj([getaliceproj(x,a).full(squeeze = True) for a in range(3)]).trans() for x in range(2)
    ]

def getBobGCLMP():
   return [
        qt.Qobj([getbobproj(x,a).full(squeeze = True) for a in range(3)]).trans() for x in range(2)
    ] 

