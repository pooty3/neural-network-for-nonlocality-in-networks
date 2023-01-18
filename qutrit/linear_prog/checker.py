from scipy.optimize import linprog

def within_bounds(vertices, point, dim):
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

print(within_bounds([[0,1], [1,0], [1,1], [0,0]], [1.0,1.0], 2))