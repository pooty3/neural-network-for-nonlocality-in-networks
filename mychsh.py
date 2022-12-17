import numpy as np
def customLoss_distr(y_pred):
    x_probs = y_pred[:,0:2]
    y_probs = y_pred[:,2:4]
    temp_start = 4
    a_probs = y_pred[:,4:6]
    b_probs = y_pred[:,6:8]

    x_probs = np.reshape(x_probs,(-1,2,1,1,1))
    y_probs = np.reshape(y_probs,(-1,1,2,1,1))
    a_probs = np.reshape(a_probs,(-1,1,1,2,1))
    b_probs = np.reshape(b_probs,(-1,1,1,1,2))

    probs = x_probs*y_probs*a_probs*b_probs
    probs = np.mean(probs,axis=0)
    #probs = np.flatten(probs)
    return probs.flatten()


lst = np.array([[0,1,0,1,1,0,0,1], [0,1,0,1,0,1,0,1]])
print(customLoss_distr(lst))

qt.identity(2) + qt.sigmaz() 
qt.bell('00').proj() + qt.identity(4)