import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import qr

import qutip as qt
import random

def get_spin_projectors(vec, N = 3):
    mats = qt.jmat((N-1.0)/2)
    spin_op = mats[0]*vec[0] + mats[1]*vec[1] + mats[2]*vec[2]
    [_, state] = spin_op.eigenstates()   

    return [st.proj() for st in state]



def get_random_unitary2(N = 3):
    return qt.rand_unitary_haar(N)

def projectors(U, N = 3):
    new_proj = [(U*qt.basis(N, i)).proj() for i in range(N)]
    return new_proj
def get_maximally_entangled_state(N = 3):
    return (1/np.sqrt(N)*sum([qt.tensor(qt.basis(N,i), qt.basis(N,i)) for i in range(N)])).proj()

def werner_state(w, N = 3):
    s1 = get_maximally_entangled_state()*w
    s2 = (1-w)/(N*N)*qt.tensor(qt.identity(N), qt.identity(N))
    return s1+s2

def getprobabilities(alice_U, bob_U, w, N = 3):
    quantum_resource = werner_state(w, N)
    alice_proj = projectors(alice_U, N)
    bob_proj = projectors(bob_U, N)
    probs = [np.abs((qt.tensor(alice_proj[i], bob_proj[j])*quantum_resource).tr()) for i in range(N) for j in range(N)]
    assert(np.abs(sum(probs) -1 ) < 0.00001)
    return probs

    
def get_spin_probs(alice_vec, bob_vec, w, N = 3):
    quantum_resource = werner_state(w, N)
    alice_proj = get_spin_projectors(alice_vec, N)
    bob_proj = get_spin_projectors(bob_vec, N)
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



def get_werner2(w):
    b1 = qt.bell_state('11')
    be = w*b1.proj()
    id = (1-w)/4*qt.tensor(qt.identity(2), qt.identity(2))
    return be + id

def getproj2(qo,val):
    return 1/2*(val*qo + qt.tensor(qt.identity(1), qt.identity(1)))


def get_prob(a,b,x,y,w):
    aa = 1 if a == 0 else -1
    bb = 1 if b == 0 else -1
    xx = qt.sigmaz() if x == 0 else qt.sigmax()
    yy = 1/np.sqrt(2) * ((qt.sigmaz() + qt.sigmax()) if y == 0 else (qt.sigmaz() - qt.sigmax()))
    xx_p = getproj2(xx,aa)
    yy_p = getproj2(yy,bb)
    pro = qt.tensor(xx_p, yy_p)
    wer = get_werner2(w)
    ##print(wer)
    ##print(pro)
    return (wer*pro).tr()

def getCHSHprob(w):
    arr = []
    for x in range(2):
        for y in range(2):
            for a in range(2):
                for b in range(2):
                    arr.append(get_prob(a,b,x,y,w))
    return arr
