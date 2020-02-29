import numpy as np
import matplotlib.pyplot as plt



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