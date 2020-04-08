"""

Introduction to Signal and Image Processing
Homework 1

Created on Mar 16, 2020
@author: Stefan weber
    
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt


def test_interp():
    # Tests the interp() function with a known input and output
    # Leads to error if test fails

    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    y = np.array([0.2, 0.4, 0.6, 0.4, 0.6, 0.8, 1.0, 1.1])
    x_new = np.array((0.5, 2.3, 3, 5.45))
    y_new_solution = np.array([0.2, 0.46, 0.6, 0.69])
    y_new_result = interp(y, x, x_new)
    np.testing.assert_almost_equal(y_new_solution, y_new_result)


def test_interp_1D():
    # Test the interp_1D() function with a known input and output
    # Leads to error if test fails

    y = np.array([0.2, 0.4, 0.6, 0.4, 0.6, 0.8, 1.0, 1.1])
    y_rescaled_solution = np.array([
        0.20000000000000001, 0.29333333333333333, 0.38666666666666671,
        0.47999999999999998, 0.57333333333333336, 0.53333333333333333,
        0.44000000000000006, 0.45333333333333331, 0.54666666666666663,
        0.64000000000000001, 0.73333333333333339, 0.82666666666666677,
        0.91999999999999993, 1.0066666666666666, 1.0533333333333335,
        1.1000000000000001
    ])
    y_rescaled_result = interp_1D(y, 2)
    np.testing.assert_almost_equal(y_rescaled_solution, y_rescaled_result)


def test_interp_2D():
    # Tests interp_2D() function with a known and unknown output
    # Leads to error if test fails

    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    matrix_scaled = np.array([[1., 1.4, 1.8, 2.2, 2.6, 3.],
                              [2., 2.4, 2.8, 3.2, 3.6, 4.],
                              [3., 3.4, 3.8, 4.2, 4.6, 5.],
                              [4., 4.4, 4.8, 5.2, 5.6, 6.]])

    result = interp_2D(matrix, 2)
    np.testing.assert_almost_equal(matrix_scaled, result)


def interp(y_vals, x_vals, x_new):
    # Computes interpolation at the given abscissas
    #
    # Inputs:
    #   x_vals: Given inputs abscissas, numpy array
    #   y_vals: Given input ordinates, numpy array
    #   x_new : New abscissas to find the respective interpolated ordinates, numpy
    #   arrays
    #
    # Outputs:
    #   y_new: Interpolated values, numpy array

    ################### PLEASE FILL IN THIS PART ###############################
    
    # initialize Array that will be filled with interpolated y values
    y_new = np.array([])

    # For loop trough all new x values
    for i in range(len(x_new)):
        
        #finding the minimum and make sure that always the left x values will be chosen
        diff =  x_new[i] - x_vals
        diff = np.where(diff >= 0, diff, np.inf)

        index = diff.argmin()

        #index = (np.abs(x_vals - x_new[i])).argmin()

        # if new x value is not in the range of the old x value 
        if x_new[i] <= np.amin(x_vals):
             y_new = np. append(y_new, y_vals[0])

        elif x_new[i] >= np.amax(x_vals):
             y_new = np. append(y_new, y_vals[-1])

        # interpolate for the new y value
        else:
            y_new = np. append(y_new, y_vals[index]+ (y_vals[index +1] - y_vals[index]) * (x_new[i]-x_vals[index])/(x_vals[index +1] - x_vals[index]))

        
    # Solution withe np.interp
    #for i in range(len(x_new)):
    #    y_new = np.append(y_new, np.interp(x_new[i], x_vals, y_vals))


    return y_new


def interp_1D(signal, scale_factor):
    # Linearly interpolates one dimensional signal by a given saling fcator
    #
    # Inputs:
    #   signal: A one dimensional signal to be samples from, numpy array
    #   scale_factor: scaling factor, float
    #
    # Outputs:
    #   signal_interp: Interpolated 1D signal, numpy array

    ################### PLEASE FILL IN THIS PART ##############################
    
    # generate old and new x values
    samplesNew = int(len(signal) * scale_factor)
    samplesOld = len(signal)
    x_vals =np.array( np.linspace(0, 1, num=samplesOld))
    x_new =np.array( np.linspace(0, 1, num=samplesNew))

    # reuse the function
    signal_interp = interp(signal, x_vals, x_new)





    return signal_interp


def interp_2D(img, scale_factor):
    # Applies bilinear interpolation using 1D linear interpolation
    # It first interpolates in one dimension and passes to the next dimension
    #
    # Inputs:
    #   img: 2D signal/image (grayscale or RGB), numpy array
    #   scale_factor: Scaling factor, float
    #
    # Outputs:
    #   img_interp: interpolated image with the expected output shape, numpy array

    ################### PLEASE FILL IN THIS PART ###############################
    
    # if it's a RGB image the length of the shape will be 3
    if len(img.shape) == 3:
        # init array for interpolation in the first axis
        img_interp1 = np.zeros(shape=(img.shape[0],int(img.shape[1]*scale_factor),3))
        # interpola through all RGB channels
        for i in range(img.shape[0]):        
            img_interp1[i,:,0] = interp_1D(img[i,:,0], scale_factor)
            img_interp1[i,:,1] = interp_1D(img[i,:,1], scale_factor)
            img_interp1[i,:,2] = interp_1D(img[i,:,2], scale_factor)

        #init final Image
        img_interp = np.zeros(shape=(int(img.shape[0]*scale_factor),int(img.shape[1]*scale_factor),3))
        # interpolate in the second axis 
        for k in range(img_interp1.shape[1]):
            img_interp[:,k,0] = interp_1D(img_interp1[:,k,0], scale_factor)
            img_interp[:,k,1] = interp_1D(img_interp1[:,k,1], scale_factor)
            img_interp[:,k,2] = interp_1D(img_interp1[:,k,2], scale_factor)

    # if it's a grayscale image there will be only one intensity channel
    else:  
        # init array for interpolation in the first axis   
        img_interp1 = np.zeros(shape=(img.shape[0],int(img.shape[1]*scale_factor)))
        # interpolate in the first axis 
        for i in range(img.shape[0]):        
            img_interp1[i,:] = interp_1D(img[i,:], scale_factor)

        # init final image 
        img_interp = np.zeros(shape=(int(img.shape[0]*scale_factor),int(img.shape[1]*scale_factor)))
        # interpolate in the second axis 
        for k in range(img_interp1.shape[1]):
            img_interp[:,k] = interp_1D(img_interp1[:,k], scale_factor)

    return img_interp


# set arguments
# filename = 'bird.jpg'
filename = 'butterfly.jpg'
scale_factor = 1.5  # Scaling factor

# Before trying to directly test the bilinear interpolation on an image, we
# test the intermediate functions to see if the functions that are coded run
# correctly and give the expected results.

print('...................................................')
print('Testing test_interp()...')
test_interp()
print('done.')

print('Testing interp_1D()....')
test_interp_1D()
print('done.')

print('Testing interp_2D()....')
test_interp_2D()
print('done.')

print('Testing bilinear interpolation of an image...')
# Read image as a matrix, get image shapes before and after interpolation
img = (plt.imread(filename)).astype('float')  # need to convert to float
in_shape = img.shape  # Input image shape

# Apply bilinear interpolation
img_int = interp_2D(img, scale_factor)
print('done.')

# Now, we save the interpolated image and show the results
print('Plotting and saving results...')
plt.figure()
plt.imshow(img_int.astype('uint8'))  # Get back to uint8 data type
filename, _ = os.path.splitext(filename)
plt.savefig('{}_rescaled.jpg'.format(filename))
plt.close()

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img.astype('uint8'))
plt.title('Original')
plt.subplot(1, 2, 2)
plt.imshow(img_int.astype('uint8'))
plt.title('Rescaled by {:2f}'.format(scale_factor))
print('Do not forget to close the plot window --- it happens:) ')
plt.show()

print('done.')
