import numpy as np
import matplotlib.pyplot as plt
import math

def get_polar(vec):
    [x,y,z] = vec
    phi = math.atan2(y, x)
    theta = math.acos(z)
    return [phi, theta]


f = open("some_data.txt", "r")
lines = f.readlines()

def dot_product(vec1, vec2):
    return sum([x*y for x,y in zip(vec1, vec2)])
class Data:
    def __init__(self, line):
        tokens = list(map(float, line.split()))
        self.lam1 = tokens[:2]
        self.lam2 = tokens[2:4]
        self.x = tokens[4:6]
        self.y = tokens[6:8]
        self.a1 = tokens[8:11]
        self.b1 = tokens[11:14]
        self.a2 = tokens[14:17]
        self.b2 = tokens[17:20]
        self.c = tokens[20]





SIZE = 000000
CC = 100000
datas = list(map(Data, lines[SIZE:SIZE + CC]))

# l1 = datas[0].lam1
# l2 = datas[0].lam2

# X = [dot_product(l1, data.y) for data in datas]
# Y = [dot_product(l2, data.y) for data in datas]
# p = [data.b2 for data in datas]
# plt.scatter(x = X, y = Y, c = p, cmap = 'viridis')
# plt.title(r'Scatterplot of $P_{B_2}(b = 0)$')
# plt.xlabel(r'$\vec{\lambda_1} \cdot \vec{y}$')
# plt.ylabel(r'$\vec{\lambda_2} \cdot \vec{y}$')

# c1 = plt.colorbar()
# c1.set_label(r'$P_{B_1}(b = 0)$')
# plt.savefig("Figure_B25.png")

def plotter(l1, l2, file, phis, thetas, p, label):
    plt.cla()
    plt.clf()
    plt.scatter(x = phis, y = thetas, c = p, cmap = 'viridis')
    plt.title("Scatterplot of " + label)
    plt.xlabel(r'$\phi$ (radians)')
    plt.ylabel(r'$\theta$ (radians)')

    [p1, t1] = l1
    [p2, t2] = l2
    plt.plot(p1, t1, marker = "*", ls = "none", ms = 20)
    plt.plot(p2, t2, marker = "X", ls = "none", ms = 20)

    c1 = plt.colorbar()
    c1.set_label(label)
    plt.savefig(file)

plotter(datas[0].lam1, datas[0].lam2, 
        "Figure_B2_0.png", 
        [data.y[0] for data in datas], 
        [data.y[1] for data in datas], 
        [data.b2[0] for data in datas],
        "P(c = 0)")
plotter(datas[0].lam1, datas[0].lam2, 
        "Figure_B2_1.png", 
        [data.y[0] for data in datas], 
        [data.y[1] for data in datas], 
        [data.b2[1] for data in datas],
        "P(c = 0)")
plotter(datas[0].lam1, datas[0].lam2, 
        "Figure_B2_2.png", 
        [data.y[0] for data in datas], 
        [data.y[1] for data in datas], 
        [data.b2[2] for data in datas],
        "P(c = 0)")