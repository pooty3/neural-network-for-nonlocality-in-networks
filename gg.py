import numpy as np
import matplotlib.pyplot as plt
arrA =[0.00363274, 0.00953818, 0.00569269, 0.00521613, 0.00546663,
       0.00992475, 0.00713688, 0.01073931, 0.01056529, 0.00879574]
arrB = [0.00905289, 0.00674761, 0.0102957 , 0.00568736, 0.00495804,
       0.0103293 , 0.00686303, 0.01658175, 0.03756639, 0.05101371]
x = np.linspace(0, 1, 10)

plt.scatter(x, arrA, marker = '*', label = '1 bit of communication', s = 100)
plt.scatter(x, arrB, marker = '+', label = 'no communication', s=100)
plt.legend()
plt.grid(True)
plt.xlabel('W')
plt.ylabel('ML Loss')
plt.title('Graph of ML Loss against W')
plt.savefig("pic.png")