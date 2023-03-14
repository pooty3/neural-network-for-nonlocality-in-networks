import distributions as db
import checker as ch
import numpy as np
import random
prob_arr = [
    [1,0,0,0,1,0,0,0,1],
    [1,0,0,0,0,1,0,1,0],
    [0,1,0,1,0,0,0,0,1],
    [0,1,0,0,0,1,1,0,0],
    [0,0,1,1,0,0,0,1,0],
    [0,0,1,0,1,0,1,0,0]
]
EPS = 1e-4
def get_noise(total_dim):
    return np.ones((total_dim,))
def convert_to_prob(str):
    return np.array([prob_arr[int(ch)] for ch in str]).flatten()/3

def get_prob(str, w):
    actual = convert_to_prob(str)*(1-w)
    noise = get_noise(len(str)*9)/9*w
    return noise + actual


#how much noise needed to go inside
def get_noise_threshhold(str, A):
    if (ch.within_bounds_A(A, list(get_prob(str, 0.0)))):
        return 0.0
    low = 0.0
    high = 1.0
    iter = 0
    while (high - low > EPS):
        iter += 1
        mid = (low + high)/2
        if (ch.within_bounds_A(A, list(get_prob(str, mid)))):
            high = mid
        else:
            low = mid
    return (low + high)/2
Acomm = ch.generate_polytope_1_bit(3,3,3,3).get_A_matrix()
Anon = ch.generate_polytope_no_comm(3,3,3,3).get_A_matrix()

def pr(val):
    return "{:.4f}".format(val)
def random_str():
    ss = ""
    for _ in range(9):
        gg = random.randint(0,5)
        ss += str(gg)
    return ss

class Debugger:
    def __init__(self, file_name):
        self.f = open(file_name, "w")
    
    def write(self, ss):
        self.f.write(ss)
        self.f.write("\n")
    

dg = Debugger("logger.txt")


for _ in range(3):
     ss = random_str()
     dg.write("string is :"+ ss)
     dg.write("Comm :"+ pr(get_noise_threshhold(ss, Acomm)))
     dg.write("No Comm :"+ pr(get_noise_threshhold(ss, Anon)))
