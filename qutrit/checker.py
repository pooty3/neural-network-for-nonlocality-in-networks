from scipy.optimize import linprog

import qutip as qt
import numpy as np
import distributions as db
import time
from pathlib import Path
import pickle

class SavedA:
    def __init__(self, A):
        self.size = A.size
        self.shape = A.shape
        self.bits = np.packbits(A, axis = None)
    def save_file(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
    def get_A(self):
        return np.unpackbits(self.bits, count = self.size).reshape(self.shape).view(bool)

def load_saved_A(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)
class Polytope:
    def __init__(self):
        self.vertices = []
        self.vertex_hashes = set()

    def gethash(self, arr):
        curtot = 0
        val = 1
        for x in arr:
            if x:
                curtot += val
            val *= 2
        return curtot
    
    def add_vertex(self, arr):
        hash = self.gethash(arr)
        if hash in self.vertex_hashes:
            return
        self.vertices.append(arr)
        self.vertex_hashes.add(hash)

    def get_A_matrix(self):

        n = len(self.vertices)
        if n == 0:
            return []
        dim = len(self.vertices[0])
        A = np.empty([dim+1, n], dtype = bool)
        for i in range(dim):
            for j in range(n):
                A[i, j] = self.vertices[j][i]

        for i in range(n):
            A[dim, i] = 1
        return A

    def get_and_save_A_matrix(self, file_path):
        n = len(self.vertices)
        if n == 0:
            return
        dim = len(self.vertices[0])
        with open(file_path, 'w') as f:
            for i in range(dim):
                for j in range(n):
                    f.write(str(self.vertices[j][i]) + ' ')
                f.write('\n')
            for j in range(n):
                f.write('1 ')
            f.write('\n')
        f.close()
    
    def save_as_pickle_A(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.get_A_matrix(), f)

    def __str__(self):
        return "polytope has" + str(len(self.vertices)) + "points :" + str(self.vertices) + "\n hashes are: " + str(self.vertex_hashes) + "\n"

def read_pickle_A(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def read_A(file_name):
    arr = []
    idx = 0
    with open(file_name) as f:
        for line in f.readlines():
            if (idx%300000 == 0):
                print(idx)
            idx += 1
            arr.append(list(map(int, line.split())))
    f.close()
    return arr

def read_polytope(txt_file):
    polytope = Polytope()
    idx = 0
    with open(txt_file) as f:
        for line in f.readlines():
            if (idx%300000 == 0):
                print(idx)
                print(len(polytope.vertices))
            idx += 1
            t = list(map(int, line.split()))
            polytope.add_vertex(t)
    return polytope


N = 3

def within_bounds_A(A, point):
    b = point.copy()
    b.append(1)
    return linprog([0 for _ in range(len(A[0]))], A_eq = A, b_eq = b).success

def within_bounds(polytope, point):
    A = polytope.get_A_matrix()
    #print(type(A))
    #print(A)
    #print(A[0])
    return within_bounds_A(A, point)



def convertToArr(x,len, base):
    currentDiv = 1
    arr = np.empty([len], dtype = int)
    for idx in range(len):
        arr[idx] = ((x//currentDiv)%base)
        currentDiv *= base
    return arr



# format will be (probs for 00, 01, ..... nn), each prob is (00, 01,.. mm)
def generate_polytope_no_comm(alice_input_count, bob_input_count, alice_output_count, bob_output_count):
    #total dimensions: (a_o*b_o)^(a_i, b_i)
    mx_alice = alice_output_count ** alice_input_count
    mx_bob = bob_output_count ** bob_input_count

    polytope = Polytope()
    dim = alice_input_count*bob_input_count*alice_output_count*bob_output_count
    for alice_res in range(mx_alice):
        alice_arr = convertToArr(alice_res, alice_input_count, alice_output_count)
        for bob_res in range(mx_bob):
            bob_arr = convertToArr(bob_res, bob_input_count, bob_output_count)
            vertex = np.empty([dim], dtype = bool)
            idx = 0
            for x in range(alice_input_count):
                for y in range(bob_input_count):
                    for a in range(alice_output_count):
                        for b in range(bob_output_count):
                            if (a == alice_arr[x] and b == bob_arr[y]):
                                vertex[idx] = 1
                            else:
                                vertex[idx] = 0
                            idx += 1
            polytope.add_vertex(vertex)
    return polytope

# format will be (probs for 00, 01, ..... nn), each prob is (00, 01,.. mm)
def generate_polytope_1_bit(alice_input_count, bob_input_count, alice_output_count, bob_output_count):
    #total dimensions: (a_o*b_o)^(a_i, b_i)
    mx_alice = alice_output_count ** alice_input_count
    mx_bob = bob_output_count ** (2*bob_input_count)
    mx_f = 2**alice_input_count
    polytope = Polytope()
    dim = alice_input_count*bob_input_count*alice_output_count*bob_output_count
    for func in range(mx_f):
        func_arr = convertToArr(func, alice_input_count, 2)
        for alice_res in range(mx_alice):
            alice_arr = convertToArr(alice_res, alice_input_count, alice_output_count)
            for bob_res in range(mx_bob):
                bob_arr = convertToArr(bob_res, 2*bob_input_count, bob_output_count)
                vertex = np.empty([dim], dtype = bool)
                idx = 0
                for x in range(alice_input_count):
                    for y in range(bob_input_count):
                        bit_sent = func_arr[x]
                        wanted_bob = bob_arr[y + (bob_input_count if bit_sent else 0)]
                        for a in range(alice_output_count):
                            for b in range(bob_output_count):
                                if (a == alice_arr[x] and b == wanted_bob):
                                    vertex[idx] = 1
                                else:
                                    vertex[idx] = 0
                                idx += 1
                polytope.add_vertex(vertex)
    print("Done here!")
    return polytope


def getCHSHthreshhold():
    polytope = generate_polytope_no_comm(2,2,2,2)
    high = 1
    low = 0
    while (abs (high - low )> 1e-9):
        mid = (high+low)/2
        prob = db.getCHSHprob(mid)
        if within_bounds(polytope, prob):
            low = mid
        else:
            high = mid
    return low

#print(getCHSHthreshhold())

def testCHSH_comm():
    polytope = generate_polytope_1_bit(2,2,2,2)
    print(polytope)
    for w in np.linspace(0,1,100):
        prob = db.getCHSHprob(w)
        print(w, ": ", within_bounds(polytope, prob))

def test_qutrit_CGLMP(communication):
    polytope = generate_polytope_1_bit(2,2, 3,3) if communication else generate_polytope_no_comm(2,2, 3,3)
    alice_U = db.getaliceGCLMP()
    bob_U = db.getBobGCLMP()
    for w in np.linspace(0,1,50):
        prob = []
        for a in range(2):
            for b in range(2):
                prob += db.getprobabilities(alice_U[a],bob_U[b], w)
        res = within_bounds(polytope, prob)
        print(w, ": ", res)


def test_qutrit(alice_settings, bob_settings, communication):
    polytope = generate_polytope_1_bit(alice_settings, bob_settings, 3,3) if communication else generate_polytope_no_comm(alice_settings, bob_settings, 3,3) 
    print("Heyo1!")
    A = polytope.get_A_matrix()
    print("Heyo!")
    TRIES = 100
    for w in np.linspace(0.9,1,3):
        success = 0
        for _ in range(TRIES):
            alice_U = [db.get_random_unitary2() for _ in range(alice_settings)]
            bob_U = [db.get_random_unitary2() for _ in range(bob_settings)]
            prob = []
            for a in range(alice_settings):
                for b in range(bob_settings):
                    prob += db.getprobabilities(alice_U[a],bob_U[b], w)
            tic = time.perf_counter()
            if (within_bounds_A(A, prob)):
                success +=1
            toc = time.perf_counter()
            print(toc-tic)
        print(w, ": ", success/TRIES)


#test_qutrit(3,3, True)
#test_qutrit(4,4, True)
#testCHSH_comm()
#test_qutrit_comm_CGLMP()
#test_qutrit_comm(3,3)

def check_violation_for_setting(alice_in, bob_in, alice_out, bob_out, filepath):
    path = Path(filepath)
    if not path.is_file():
        poly = generate_polytope_1_bit(alice_in, bob_in, alice_out, bob_out).get_A_matrix()
        SavedA(poly).save_file(filepath)
    A = load_saved_A(filepath).get_A()
    print("A length: ", len(A))
    print("A row length: ", len(A[0]))
    fast_rej_count = 0
    A_no_comm = generate_polytope_no_comm(alice_in, bob_in, alice_out, bob_out).get_A_matrix()
    for tr in range(100):
        alice_U = [db.get_random_unitary2(N = alice_out) for _ in range(alice_in)]
        bob_U = [db.get_random_unitary2(N = bob_out) for _ in range(bob_in)]
        prob = []
        for a in range(alice_in):
            for b in range(bob_in):
                prob += db.getprobabilities(alice_U[a],bob_U[b], 1, N = alice_out)
        
        print(tr)
        tic = time.perf_counter()
        if (within_bounds_A(A_no_comm, prob)):
            fast_rej_count += 1 
            print("Fast reject!", fast_rej_count, "otu of ", tr+1, "percentage of ", float(fast_rej_count)/(tr+1))
            toc = time.perf_counter()
            print(toc - tic)
            continue
        if (not within_bounds_A(A, prob)):
            print("Bob: ", bob_U)
            print("Alice: ", alice_U)
            break
        toc = time.perf_counter()
        print(toc - tic)

#check_violation_for_setting(3,3,3,3, "3333.obj")
#check_violation_for_setting(3,4,3,3, "3433.obj") #11
#check_violation_for_setting(4,3,3,3, "4333.obj") #9