import numpy as np
import random

def logistic_map(x, r):
    y = r*x*(1-x)
    return y
    
def iterate_f(my_iter,x,r):
    history = []
    for iters in range(my_iter):
        x = logistic_map(x, r)
        history.append(x)
    return np.array(history)
    

