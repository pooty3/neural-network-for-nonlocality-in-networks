import numpy as np
import matplotlib.pyplot as plt
import math

def get_polar(vec):
    [x,y,z] = vec
    phi = math.atan2(y, x)
    theta = math.acos(z)
    return [phi, theta]

<<<<<<< HEAD
def get_cart(vv):
    theta = vv[1]
    phi = vv[0]
    z = math.cos(theta)
    x = math.sin(theta)*math.cos(phi)
    y = math.sin(theta)*math.sin(phi)
    return [x,y,z]
=======
>>>>>>> c6e660feeeb2df5c67f6dcbb70a81427bb23dab7

f = open("some_data.txt", "r")
lines = f.readlines()

def dot_product(vec1, vec2):
    return sum([x*y for x,y in zip(vec1, vec2)])
class Data:
    def __init__(self, line):
        tokens = list(map(float, line.split()))
<<<<<<< HEAD
        self.lam1 = get_cart(tokens[:2])
        self.lam2 = get_cart(tokens[2:4])
        self.x = get_cart(tokens[4:6])
        self.y = get_cart(tokens[6:8])
=======
        self.lam1 = tokens[:2]
        self.lam2 = tokens[2:4]
        self.x = tokens[4:6]
        self.y = tokens[6:8]
>>>>>>> c6e660feeeb2df5c67f6dcbb70a81427bb23dab7
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

<<<<<<< HEAD
# def plotter(l1, l2, file, phis, thetas, p, label):
#     plt.cla()
#     plt.clf()
#     plt.scatter(x = phis, y = thetas, c = p, cmap = 'viridis')
#     plt.title("Scatterplot of " + label)
#     plt.xlabel(r'$\phi$ (radians)')
#     plt.ylabel(r'$\theta$ (radians)')

#     [p1, t1] = l1
#     [p2, t2] = l2
#     plt.plot(p1, t1, marker = "*", ls = "none", ms = 20)
#     plt.plot(p2, t2, marker = "X", ls = "none", ms = 20)

#     c1 = plt.colorbar()
#     c1.set_label(label)
#     plt.savefig(file)

def draw_lam(ax, l, m, col, text):
    XX = l[0]*1.3
    YY = l[1]*1.3
    ZZ = l[2]*1.3
    ax.plot([0,XX], [0,YY], [0,ZZ],ms = 20,marker = m, c= col)
    ax.text(XX*1.1,YY*1.1,ZZ*1.1, text, l)

def plotter(l1, l2, XXX, TOPLOT, title, filename):
    plt.cla()
    plt.clf()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    X = [x[0] for x in XXX]
    Y = [x[1] for x in XXX]
    Z = [x[2] for x in XXX]
    img = ax.scatter(X, Y, Z, marker='s',
                  c=TOPLOT)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim3d(-1.2, 1.2)
    ax.set_ylim3d(-1.2, 1.2)
    ax.set_zlim3d(-1.2, 1.2)
    draw_lam(ax, l1, '*', "blue", r'$\lambda_1$')
    draw_lam(ax, l2, '^', "red", r'$\lambda_2$')
#ax.plot([0,l1[0]*1.3], [0,l1[1]*1.3], [0,l1[2]*1.3],ms = 20,marker = "*", c= "blue", label = r'$\lambda_1$')
#ax.plot([0,l2[0]*1.3], [0,l2[1]*1.3], [0,l2[2]*1.3],ms = 20,marker = "+", c= "red", label = r'$\lambda_2$')
#ax.plot(l2[0]*1.1, l2[1]*1.1, l2[0]*1.1, marker = "^", ms = 40, c= "red")
   # bar = fig.colorbar(img, shrink = 0.4)
   # bar.set_label(ctitle)
    ax.view_init(elev=19, azim=18)
    plt.savefig(filename, bbox_inches='tight')
    #plt.show()

plotter(datas[0].lam1, datas[0].lam2, [data.x for data in datas],
        [data.a1 for data in datas], "Alice 1 output", "A1out.png")

plotter(datas[0].lam1, datas[0].lam2, [data.x for data in datas],
        [data.a2 for data in datas], "Alice 2 output", "A2out.png")

plotter(datas[0].lam1, datas[0].lam2, [data.y for data in datas],
        [data.b1 for data in datas], "Bob 1 output", "B1out.png")

plotter(datas[0].lam1, datas[0].lam2, [data.y for data in datas],
        [data.b2 for data in datas], "Bob 2 output", "B2out.png")

#plt.savefig(filename, bbox_inches='tight')

# print([data.x[0] for data in datas])
# print([data.x[1] for data in datas])

# print([data.a1 for data in datas])


# plotter(datas[0].lam1, datas[0].lam2, 
#         "Figure_B2_0.png", 
#         [data.y[0] for data in datas], 
#         [data.y[1] for data in datas], 
#         [data.b2[0] for data in datas],
#         "P(c = 0)")
# plotter(datas[0].lam1, datas[0].lam2, 
#         "Figure_B2_1.png", 
#         [data.y[0] for data in datas], 
#         [data.y[1] for data in datas], 
#         [data.b2[1] for data in datas],
#         "P(c = 0)")
# plotter(datas[0].lam1, datas[0].lam2, 
#         "Figure_B2_2.png", 
#         [data.y[0] for data in datas], 
#         [data.y[1] for data in datas], 
#         [data.b2[2] for data in datas],
#         "P(c = 0)")
=======
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
>>>>>>> c6e660feeeb2df5c67f6dcbb70a81427bb23dab7
