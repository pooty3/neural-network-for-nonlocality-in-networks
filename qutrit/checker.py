from scipy.optimize import linprog

import qutip as qt
import numpy as np
import distributions as db
import time
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
    return (wer*pro).tr()

def getCHSHprob(w):
    arr = []
    for x in range(2):
        for y in range(2):
            for a in range(2):
                for b in range(2):
                    arr.append(get_prob(a,b,x,y,w))
    return arr


N = 3


def within_bounds(vertices, point, dim):
    print(len(vertices))
    n = len(vertices)
    b = point.copy()
    b.append(1)
    A = []
    for i in range(dim):
        cur = []
        for ver in vertices:
            cur.append(ver[i])
        A.append(cur)
    A.append([1 for _ in range(n)])   
    res = linprog([0 for _ in range(n)], A_eq = A, b_eq = b)
    return res.success

def convertToArr(x,len, base):
    currentDiv = 1
    arr = []
    for _ in range(len):
        arr.append((x//currentDiv)%base)
        currentDiv *= base
    return arr


# format will be (probs for 00, 01, ..... nn), each prob is (00, 01,.. mm)
def generate_polytope_no_comm(alice_input_count, bob_input_count, alice_output_count, bob_output_count):
    #total dimensions: (a_o*b_o)^(a_i, b_i)
    mx_alice = alice_output_count ** alice_input_count
    mx_bob = bob_output_count ** bob_input_count
    vertices = []
    for alice_res in range(mx_alice):
        alice_arr = convertToArr(alice_res, alice_input_count, alice_output_count)
        for bob_res in range(mx_bob):
            bob_arr = convertToArr(bob_res, bob_input_count, bob_output_count)
            vertex = []
            for x in range(alice_input_count):
                for y in range(bob_input_count):
                    for a in range(alice_output_count):
                        for b in range(bob_output_count):
                            if (a == alice_arr[x] and b == bob_arr[y]):
                                vertex.append(1)
                            else:
                                vertex.append(0)
            vertices.append(vertex)
    return vertices

# format will be (probs for 00, 01, ..... nn), each prob is (00, 01,.. mm)
def generate_polytope_1_bit(alice_input_count, bob_input_count, alice_output_count, bob_output_count):
    #total dimensions: (a_o*b_o)^(a_i, b_i)
    mx_alice = alice_output_count ** alice_input_count
    mx_bob = bob_output_count ** (2*bob_input_count)
    mx_f = 2**alice_input_count

    vertices = []
    for func in range(mx_f):
        func_arr = convertToArr(func, alice_input_count, 2)
        for alice_res in range(mx_alice):
            alice_arr = convertToArr(alice_res, alice_input_count, alice_output_count)
            for bob_res in range(mx_bob):
                bob_arr = convertToArr(bob_res, 2*bob_input_count, bob_output_count)
                vertex = []
                print(func, alice_res, bob_res)
                for x in range(alice_input_count):
                    for y in range(bob_input_count):
                        bit_sent = func_arr[x]
                        wanted_bob = bob_arr[y + (bob_input_count if bit_sent else 0)]
                        for a in range(alice_output_count):
                            for b in range(bob_output_count):
                                if (a == alice_arr[x] and b == wanted_bob):
                                    vertex.append(1)
                                else:
                                    vertex.append(0)
                vertices.append(vertex)
    print(len(vertices))
    return vertices




#for w in np.linspace(0,1, 100):
 #   prob = getCHSHprob(w)
  #  print(w, ': ', within_bounds(polytope, prob, len(prob)))

def getCHSHthreshhold():
    polytope = generate_polytope_no_comm(2,2,2,2)
    high = 1
    low = 0
    while (abs (high - low )> 1e-9):
        mid = (high+low)/2
        prob = getCHSHprob(mid)
        if (within_bounds(polytope, prob, len(prob))):
            low = mid
        else:
            high = mid
    return low


def testCHSH_comm():
    polytope = generate_polytope_1_bit(2,2,2,2)
    for w in np.linspace(0,1,100):
        prob = getCHSHprob(w)
        print(w, ": ", within_bounds(polytope, prob, len(prob)))

def test_qutrit_comm_CGLMP():
    polytope = generate_polytope_1_bit(2,2, 3,3)
    alice_U = db.getaliceGCLMP()
    bob_U = db.getBobGCLMP()
    for w in np.linspace(0,1,50):
        prob = []
        for a in range(2):
            for b in range(2):
                prob += db.getprobabilities(alice_U[a],bob_U[b], w)
        res = within_bounds(polytope, prob, len(prob))
        print(w, ": ", res)



def test_qutrit_no_comm(alice_settings, bob_settings):
    polytope = generate_polytope_no_comm(alice_settings, bob_settings, 3,3)
    TRIES = 25
    for w in np.linspace(0.69,1,10):
        success = 0
        for _ in range(TRIES):
            alice_U = [db.get_random_unitary2() for _ in range(alice_settings)]
            bob_U = [db.get_random_unitary2() for _ in range(bob_settings)]
            prob = []
            for a in range(alice_settings):
                for b in range(bob_settings):
                    prob += db.getprobabilities(alice_U[a],bob_U[b], w)
            tic = time.perf_counter()
            if (within_bounds(polytope, prob, len(prob))):
                success +=1
            toc = time.perf_counter()
            print(toc-tic)
        print(w, ": ", success/TRIES)


def test_qutrit_comm(alice_settings, bob_settings):
    polytope = generate_polytope_1_bit(alice_settings, bob_settings, 3,3)
    print("hi")
    print(len(polytope))
    TRIES = 10
    for w in np.linspace(0.69,1,10):
        success = 0
        for _ in range(TRIES):
            alice_U = [db.get_random_unitary2() for _ in range(alice_settings)]
            bob_U = [db.get_random_unitary2() for _ in range(bob_settings)]
            prob = []
            for a in range(alice_settings):
                for b in range(bob_settings):
                    prob += db.getprobabilities(alice_U[a],bob_U[b], w)
            tic = time.perf_counter()
            if (within_bounds(polytope, prob, len(prob))):
                success +=1
            toc = time.perf_counter()
            print(toc-tic)
        print(w, ": ", success/TRIES)



def test_qutrit_no_comm_CGLMP():
    polytope = generate_polytope_no_comm(2,2, 3,3)
    alice_U = db.getaliceGCLMP()
    bob_U = db.getBobGCLMP()
    for w in np.linspace(0,1,50):
        prob = []
        for a in range(2):
            for b in range(2):
                prob += db.getprobabilities(alice_U[a],bob_U[b], w)
        res = within_bounds(polytope, prob, len(prob))
        print(w, ": ", res)




#print(within_bounds([[0,1], [1,0], [1,1], [0,0]], [1.0,1.0], 2))
#test_qutrit_no_comm-test_qutrit_no_comm(3,3)
#testCHSH_comm()
#test_qutrit_comm_CGLMP()
test_qutrit_comm(3,3)