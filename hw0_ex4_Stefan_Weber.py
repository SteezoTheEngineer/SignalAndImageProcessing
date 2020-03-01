import numpy as np
import matplotlib.pyplot as plt

# Define two arrays x, y of length 100, and assign random values to them
x = np.around(np.random.rand(3)*10, decimals=1)
y = np.around(np.random.rand(3)*10, decimals=1)

# Write a function rmse(a,b) that calculates the RMSE (root mean square error) between two 1-d arrays of size N.
def RMSE(a, b):
    return np.sqrt(np.mean(np.square(a-b)))

# Calculate the RMSE between arrays x and y.
sol = RMSE(x, y)

# Check your function through the degenerate case of the RMSE between array x and itself.
# Should be 0 because all differences are 0. Checked it is 0.
solSelfe = RMSE(x,x)

# Define an array with an offset of 2 from the values of x. What do you expect the RMSE to be? Conrm by using your function.
# Should be 2 because all differences are 2. Checked it is 2.
solOffSet = RMSE(x+2, x)

# Assume you have the function of exercise 0.1, 
# for x element of [0; 100] and you want to approximate it with the line g(x) = a * x, where a = 0.3.
#Plot the two lines on the same figure, and visually check if this is a good approximation.
xp = np.linspace(0, 100, 1000)
yf = np.cos(xp*np.exp(-xp/100))+np.sqrt(xp)+1

plt.plot(xp, yf, label="Function of exercise 0.1")
plt.plot(xp, 0.3 * xp, label="Approximation g(x)=a*x")
plt.legend(loc = 'upper left')
plt.show()

# Calculate the RMSE between f(x) and g(x) for the given range.
solFunciton = RMSE(yf, 0.3 * xp)

# Try to tune the parameter a of the function g(x) so that you achieve a lower error (hint:
# try different values of a in an interval that makes sense for you, e.g. a 2 [0; 0:5]). Plot
# the RMSE for the different values of a and use the numpy.min() function to choose the
# optimal a with respect to the RMSE. Print the minimum RMSE value, and the value
# of a for which it was achieved.
a = np.linspace(0, 0.5, 100)

yg = np.outer(xp, a)

err = np.array([])
for x in range(0, yg.shape[1]):
    err = np.append(err, RMSE(yf, yg[:,x]))

print("The minimum error is: %f and it occures when the slope is set to a = %f " %(np.min(err), a[np.argmin(err)]))
plt.plot(a, err)
plt.show()



