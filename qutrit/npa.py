import distributions as db
import checker as ch
import numpy as np
from pathlib import Path
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

def retreive(alice_in, bob_in, alice_out, bob_out, filename, comm):
    if not Path(filename).is_file():
        A = (ch.generate_polytope_1_bit(alice_in, bob_in, alice_out, bob_out) if comm else  ch.generate_polytope_no_comm(alice_in, bob_in, alice_out, bob_out)).get_A_matrix()
        ch.SavedA(A).save_file(filename)
        return A
    else:
        return ch.load_saved_A(filename).get_A()
    

Acomm = retreive(4,3,3,3, "4333_1npa.obj", True)
Anon = retreive(4,3,3,3, "4333_0npa.obj", False)
#Acomm = ch.generate_polytope_1_bit(3,3,3,3).get_A_matrix()
#Anon = ch.generate_polytope_no_comm(3,3,3,3).get_A_matrix()

def pr(val):
    return "{:.4f}".format(val)
def random_str(sz = 9):
    ss = ""
    for _ in range(sz):
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

# for _ in range(3):
#      ss = random_str()
#      dg.write("string is :"+ ss)
#      dg.write("Comm :"+ pr(get_noise_threshhold(ss, Acomm)))
#      dg.write("No Comm :"+ pr(get_noise_threshhold(ss, Anon)))
LINES = 50

f = open("4333seq.txt", "w")
ss = set()
tr = 0
dups = 0
for _ in range(LINES):
    tr += 1
    print(tr)
    s1 = random_str(sz = 12)
    print(s1)
    p1 = get_noise_threshhold(s1, Anon)
    p2 = get_noise_threshhold(s1, Acomm)
    if (p1,p2) in ss:
        dups += 1
        print(s1, " is dup! Dup count: ", dups, " of ", tr)
    else:
        ss.add((p1,p2))
        f.write(s1)
        f.write(' ')
        f.write(pr(p1))
        f.write(' ')
        f.write(pr(p2))
        f.write('\n')