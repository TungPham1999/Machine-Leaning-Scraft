from __future__ import division, print_function, unicode_literals
import math
import numpy as np 
import matplotlib.pyplot as plt 

# equation
# x^2 + 5sinx = f(x)
# derivation : f(x)' = 2x + 5 cos(x)
def grad(x): 
    return 2*x + 5*np.cos(x)
# cacular equation

def cost(x):
    return x**2 + 5*np.sin(x)
# check cacular equation is it true or not and value of equation is it true or not

def myGD(eta, point):
    x = [point]
    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x[-1])) < 1e-3:
            break
        x.append(x_new)
    return (x, it)
# gradient descent with start is learing rate and end is starting point

(x1, it1) = myGD(.1, -5)
(x2, it2) = myGD(.1, 5)
print('Solution x1 = %f, cost = %f, obtained after %d iterations'%(x1[-1], cost(x1[-1]), it1))
print('Solution x2 = %f, cost = %f, obtained after %d iterations'%(x2[-1], cost(x2[-1]), it2))