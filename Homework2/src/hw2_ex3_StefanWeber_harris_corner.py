""" 3 Corner detection """

# Imports
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
from scipy.signal import convolve2d, convolve
from skimage import color, io
import pdb
from scipy.ndimage import rotate
from scipy.misc import imresize

# Load the image, convert to float and grayscale
img = io.imread('chessboard.jpg')

# Copy of the function from Ex 1.6
def gauss1d(sigma, filter_length=11):

    if filter_length % 2 == 0:
        filter_length += 1
    dx = filter_length/2
    gauss_filter = np.linspace(-dx,dx,filter_length)
    gauss_filter = np.exp(-1*(np.square(gauss_filter))/(2*np.square(sigma)))

    gauss_filter = np.divide(gauss_filter, np.sum(gauss_filter))
    gauss_filter = np.array([gauss_filter])

    return gauss_filter
    




# 3.1
# Write a function myharris(image) which computes the harris corner for each pixel in the image. The function should return the R
# response at each location of the image.
# HINT: You may have to play with different parameters to have appropriate R maps.
# Try Gaussian smoothing with sigma=0.2, Gradient summing over a 5x5 region around each pixel and k = 0.1.)
def myharris(image, w_size, sigma, k):
    # This function computes the harris corner for each pixel in the image
    # INPUTS
    # @image    : a 2-D image as a numpy array
    # @w_size   : an integer denoting the size of the window over which the gradients will be summed
    # sigma     : gaussian smoothing sigma parameter
    # k         : harris corner constant
    # OUTPUTS
    # @R        : 2-D numpy array of same size as image, containing the R response for each image location

    # Generating a two dimensional filter to calculate the derivatives in x and y direction
    dx = np.array((-1, 0, 1))[np.newaxis, :]
    dy = np.transpose(dx)
    
    # Combine the derivatives filter with a gaussian smooting to reduce noise
    gdx = convolve(gauss1d(sigma), dx)
    gdy = convolve(gauss1d(sigma).T, dy)
    
    # Compute the convolution with the derivative smooting filter
    Ixx = convolve(image, gdx, mode='same')
    Iyy = convolve(image, gdy, mode='same')
    
    #Compute product of derivatives at each pixel
    Ixy = Ixx * Iyy
    Ixx **= 2
    Iyy **= 2

    #Define the window
    window = np.ones((w_size, w_size))

    #Compute sums of products of derivatives at each pixel
    Sxx = convolve(Ixx, window, mode='same')
    Syy = convolve(Iyy, window, mode='same')
    Sxy = convolve(Ixy, window, mode='same')

    #Compute the det of the matrix
    detA = Sxx * Syy - Sxy ** 2
    #Compute the trace of the matrix
    traceA = Sxx + Syy

    #compute the corner response R        
    R = detA - k * traceA ** 2

    

    return R


 # 3.2
# Evaluate myharris on the image
R = myharris(img, 5, 0.2, 0.1)
plt.imshow(R)
plt.colorbar()
plt.show()


# 3.3
# Repeat with rotated image by 45 degrees
# HINT: Use scipy.ndimage.rotate() function
R_rotated = myharris(rotate(img, 45), 5, 0.2, 0.1)  
plt.imshow(R_rotated)
plt.colorbar()
plt.show()


# 3.4
# Repeat with downscaled image by a factor of half
# HINT: Use scipy.misc.imresize() function
R_scaled =  myharris(imresize(img, 0.5), 5, 0.2, 0.1)  
plt.imshow(R_scaled)
plt.colorbar()
plt.show()