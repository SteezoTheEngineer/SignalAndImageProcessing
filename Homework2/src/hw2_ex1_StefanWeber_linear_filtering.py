""" 1 Linear filtering """

# Imports
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
import time
import pdb

img = plt.imread('cat.jpg').astype(np.float32)

plt.imshow(img)
plt.axis('off')
plt.title('original image')
plt.show()

# 1.1
def boxfilter(n):
    # this function returns a box filter of size nxn
    box_filter = 1/np.square(n) * np.ones((n,n))

    return box_filter

# 1.2
# Implement full convolution
def myconv2(image, filt):
    # This function performs a 2D convolution between image and filt, image being a 2D image. This
    # function should return the result of a 2D convolution of these two images. DO
    # NOT USE THE BUILT IN SCIPY CONVOLVE within this function. You should code your own version of the
    # convolution, valid for both 2D and 1D filters.
    # INPUTS
    # @ image         : 2D image, as numpy array, size mxn
    # @ filt          : 1D or 2D filter of size kxl
    # OUTPUTS
    # img_filtered    : 2D filtered image, of size (m+k-1)x(n+l-1)

    # if the filter is a 1d array generate a 2d array
    if len(filt.shape) == 1:
        filt = np.array([filt])

    # Flipping of the filter to compute a convolution
    filt = np.flip(filt)
    dx = int(filt.shape[0]/2)
    dy = int(filt.shape[1]/2)

    # Gerentat a template for the filtered image an pad the image
    filtered_img = np.zeros(image.shape)
    filtered_img = np.pad(filtered_img, ((dx, dx),(dy, dy)), 'constant', constant_values=0)
    filtered_img_tmp = np.pad(image, ((2*dx, 2*dx),(2*dy, 2*dy)), 'constant', constant_values=0)

    # go trough evrey pixel of the image
    for i in range(filtered_img.shape[0]):
        for k in range(filtered_img.shape[1]):
            # difine the part of the image to multiply element wise and sum all up
            part = filtered_img_tmp[i:i+filt.shape[0],k:k+filt.shape[1]]
            filtered_img[i,k] = np.sum(np.multiply(part, filt))

    return filtered_img


# 1.3
# create a boxfilter of size 11 and convolve this filter with your image - show the result
bsize = 11

filt = boxfilter(bsize)
img_filt = myconv2(img, filt)

plt.imshow(img_filt)
plt.axis('off')
plt.title('Filterd image')
plt.show()

# 1.4
# create a function returning a 1D gaussian kernel
def gauss1d(sigma, filter_length=11):
    # INPUTS
    # @ sigma         : sigma of gaussian distribution
    # @ filter_length : integer denoting the filter length
    # OUTPUTS
    # @ gauss_filter  : 1D gaussian filter

    # make sure that the filter is odd 
    if filter_length % 2 == 0:
        filter_length += 1
    # define the filter boundaries
    dx = filter_length/2
    # generate equally distant x-values ​​and declare Gauss
    gauss_filter = np.linspace(-dx,dx,filter_length)
    gauss_filter = np.exp(-1*(np.square(gauss_filter))/(2*np.square(sigma)))

    # Normalize the filter and make a 2d array
    gauss_filter = np.divide(gauss_filter, np.sum(gauss_filter))
    gauss_filter = np.array([gauss_filter])

    return gauss_filter



# 1.5
# create a function returning a 2D gaussian kernel
def gauss2d(sigma, filter_size=11):
    # INPUTS
    # @ sigma         : sigma of gaussian distribution
    # @ filter_size   : integer denoting the filter size
    # OUTPUTS
    # @ gauss2d_filter  : 2D gaussian filter

    # Reuse guss1d and myconv2
    filt = gauss1d(sigma,filter_size)
    gauss2d_filter = myconv2(filt, filt.T)
    # Normalize the filter
    gauss2d_filter = np.divide(gauss2d_filter, np.sum(gauss2d_filter))


    return gauss2d_filter

# Display a plot using sigma = 3
sigma = 3

gauss_filt = gauss2d(sigma)
plt.imshow(gauss_filt)
plt.axis('off')
plt.title('Gauss filter')
plt.show()


# 1.6
# Convoltion with gaussian filter
def gconv(image, sigma):
    # INPUTS
    # image           : 2d image
    # @ sigma         : sigma of gaussian distribution
    # OUTPUTS
    # @ img_filtered  : filtered image with gaussian filter

    filt = gauss2d(sigma)

    img_filtered = myconv2(image, filt)

    return img_filtered


# run your gconv on the image for sigma=3 and display the result
sigma = 3

img_filt_gauss = gconv(img, sigma)

plt.imshow(img_filt_gauss)
plt.axis('off')
plt.title('Gauss filtered image')
plt.show()


# 1.7
# Convolution with a 2D Gaussian filter is not the most efficient way
# to perform Gaussian convolution with an image. In a few sentences, explain how
# this could be implemented more efficiently and why this would be faster.
#
# HINT: How can we use 1D Gaussians?

### your explanation should go here ###
## More operations are necessary if a 2d convoltuion is done .
## 1d = 2 * filter_size * n_pixels
## 2d = filter_size^2 * n_pixels

# 1.8
# Computation time vs filter size experiment
size_range = np.arange(3, 103, 5)
t1d = []
t2d = []
for size in size_range:

    filt = gauss1d(3,size)
    # Time computation Time for a 1d Convolution
    t = time.time()
    img_tmp = myconv2(img, filt)
    img_tmp = myconv2(img_tmp, filt.T)
    t1d.append(time.time()-t)

    filt = gauss2d(3,size)
    # Time computation Time for a 2d Convolution
    t = time.time()
    img_tmp = myconv2(img, filt)
    t2d.append(time.time()-t)





# plot the comparison of the time needed for each of the two convolution cases
plt.plot(size_range, t1d, label='1D filtering')
plt.plot(size_range, t2d, label='2D filtering')
plt.xlabel('Filter size')
plt.ylabel('Computation time')
plt.legend(loc=0)
plt.show() 
