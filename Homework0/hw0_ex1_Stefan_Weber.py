import numpy as np
import matplotlib.pyplot as plt


# You are given the following function:f(x) = cos(x * exp(-x/100) +sqrt(x) + 1:
# Create a function ex_0(a,b,c) that plots f(x), where x are c equally spaced numbers in the range of [a; b]. 
# If a and b are non-negative numbers the function returns 0 and plots f(x),
# otherwise it returns -1 and does not plot anything.
def ex_0(a, b, c):
    if a>=0 & b>=0:
        x = np.linspace(a, b, c)
        y = np.cos(x*np.exp(-x/100))+np.sqrt(x)+1
        plt.plot(x,y)
        plt.show()
        print(0)
        return 0
    else:
        print(-1)
        return -1


ex_0(0, 100, 1000)