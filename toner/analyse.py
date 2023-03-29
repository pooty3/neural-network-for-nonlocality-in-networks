import numpy as np
import matplotlib.pyplot as plt
import math

def get_polar(vec):
    [x,y,z] = vec
    phi = math.atan2(y, x)
    theta = math.acos(z)
    return [phi, theta]


f = open("data3.txt", "r")
lines = f.readlines()

def dot_product(vec1, vec2):
    return sum([x*y for x,y in zip(vec1, vec2)])
class Data:
    def __init__(self, line):
        tokens = list(map(float, line.split()))
        self.lam1 = tokens[:3]
        self.lam2 = tokens[3:6]
        self.x = tokens[6:9]
        self.y = tokens[9:12]
        self.a1 = tokens[12]
        self.a2 = tokens[14]
        self.b1 = tokens[16]
        self.b2 = tokens[18]
        self.c = tokens[20]





SIZE = 500000
CC = 100000
datas = list(map(Data, lines[SIZE:SIZE + CC]))

l1 = datas[0].lam1
l2 = datas[0].lam2

X = [dot_product(l1, data.y) for data in datas]
Y = [dot_product(l2, data.y) for data in datas]
p = [data.b2 for data in datas]
plt.scatter(x = X, y = Y, c = p, cmap = 'viridis')
plt.title(r'Scatterplot of $P_{B_2}(b = 0)$')
plt.xlabel(r'$\vec{\lambda_1} \cdot \vec{y}$')
plt.ylabel(r'$\vec{\lambda_2} \cdot \vec{y}$')

c1 = plt.colorbar()
c1.set_label(r'$P_{B_1}(b = 0)$')
plt.savefig("Figure_B25.png")

    
# phis = [get_polar(data.x)[0] for data in datas]
# thetas = [get_polar(data.x)[1] for data in datas]
# p = [data.c for data in datas]

# plt.scatter(x = phis, y = thetas, c = p, cmap = 'viridis')
# plt.title("Scatterplot of P(c = 0)")
# plt.xlabel(r'$\phi$ (radians)')
# plt.ylabel(r'$\theta$ (radians)')

# [p1, t1] = get_polar(datas[0].lam1)
# [p2, t2] = get_polar(datas[0].lam2)
# plt.plot(p1, t1, marker = "*", ls = "none", ms = 20)
# plt.plot(p2, t2, marker = "X", ls = "none", ms = 20)
# c1 = plt.colorbar()
# c1.set_label("P(c = 0)")
# plt.savefig("Figure_Cflat.png")
