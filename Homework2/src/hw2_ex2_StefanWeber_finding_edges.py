""" 2 Finding edges """

import numpy as np
from skimage import color, io
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
import pdb



# load image
img = io.imread('bird.jpg')

### copy functions myconv2, gauss1d, gauss2d and gconv from exercise 1 ###
def myconv2(image, filt):

    if len(filt.shape) == 1:
        filt = np.array([filt])

    filt = np.flip(filt)
    
    dx = int(filt.shape[0]/2)
    dy = int(filt.shape[1]/2)

    filtered_img = np.zeros(image.shape)
    filtered_img = np.pad(filtered_img, ((dx, dx),(dy, dy)), 'constant', constant_values=0)
    filtered_img_tmp = np.pad(image, ((2*dx, 2*dx),(2*dy, 2*dy)), 'constant', constant_values=0)

    for i in range(filtered_img.shape[0]):
        for k in range(filtered_img.shape[1]):
            part = filtered_img_tmp[i:i+filt.shape[0],k:k+filt.shape[1]]
            filtered_img[i,k] = np.sum(np.multiply(part, filt))

    return filtered_img

def gauss1d(sigma, filter_length=11):
    
    if filter_length % 2 == 0:
        filter_length += 1
    
    dx = filter_length/2
    
    gauss_filter = np.linspace(-dx,dx,filter_length)
    gauss_filter = np.exp(-1*(np.square(gauss_filter))/(2*np.square(sigma)))

    gauss_filter = np.divide(gauss_filter, np.sum(gauss_filter))
    gauss_filter = np.array([gauss_filter])

    return gauss_filter



def gauss2d(sigma, filter_size=11):

    filt = gauss1d(sigma,filter_size)
    gauss2d_filter = myconv2(filt, filt.T)
    gauss2d_filter = np.divide(gauss2d_filter, np.sum(gauss2d_filter))


    return gauss2d_filter
### end of copy

# 2.1
# Gradients
# define a derivative operator
dx = np.array((-1, 0, 1))[np.newaxis, :]
print(dx.shape)
dy = np.transpose(dx)
print(dy.shape)


# convolve derivative operator with a 1d gaussian filter with sigma = 1
# You should end up with 2 1d edge filters,  one identifying edges in the x direction, and
# the other in the y direction
sigma = 1
#Convolve 1D Gaussianfilter with dx and dy
gdx = myconv2(gauss1d(sigma), dx)
gdy = myconv2(gauss1d(sigma).T, dy)


# 2.2
# Gradient Edge Magnitude Map
def create_edge_magn_image(image, dx, dy):
    # this function created an edge magnitude map of an image
    # for every pixel in the image, it assigns the magnitude of gradients
    # INPUTS
    # @image  : a 2D image
    # @gdx     : gradient along x axis
    # @gdy     : geadient along y axis
    # OUTPUTS
    # @ grad_mag_image  : 2d image same size as image, with the magnitude of gradients in every pixel
    # @grad_dir_image   : 2d image same size as image, with the direcrion of gradients in every pixel

    # Compute derivatives in x and y direction
    grad_mag_image_dx = myconv2(image,dx)
    grad_mag_image_dy = myconv2(image, dy)

    # Make sure that the image has to originall size
    grad_mag_image_dx = grad_mag_image_dx[:, 1:-1]
    grad_mag_image_dy = grad_mag_image_dy[1:-1, :]

    # Compute the the magnitude of gradients and the direction
    grad_mag_image = np.sqrt(np.add(np.square(grad_mag_image_dx), np.square(grad_mag_image_dy)))
    grad_dir_image = np.arctan2(grad_mag_image_dy, grad_mag_image_dx)


    return grad_mag_image, grad_dir_image


# create an edge magnitude image using the derivative operator
img_edge_mag, img_edge_dir = create_edge_magn_image(img, dx, dy)

# show all together
plt.subplot(121)
plt.imshow(img)
plt.axis('off')
plt.title('Original image')
plt.subplot(122)
plt.imshow(img_edge_mag)
plt.axis('off')
plt.title('Edge magnitude map')
plt.show() 

# 2.3
# Edge images of particular directions
def make_edge_map(image, dx, dy):
    # INPUTS
    # @image        : a 2D image
    # @dx          : gradient along x axis
    # @dy          : geadient along y axis
    # OUTPUTS:
    # @ edge maps   : a 3D array of shape (image.shape, 4) containing the edge maps on 4 orientations

    # difine a threshold for the magnitude 
    threshold = 30

    # generate a 3 dimensional template for the edge maps 
    edge_maps = np.zeros((image.shape[0], image.shape[1], 4))

    # Compute the the magnitude of gradients and the direction
    img_edge_mag, img_edge_dir = create_edge_magn_image(image, dx, dy)

    # Calculate the boundaries of the directions 
    dir_low = (np.arange(0,4)*2*np.pi-np.pi)/4
    dir_up = (np.arange(0,4)*2*np.pi+np.pi)/4

    # Set the range from -pi to pi to 0 to 2pi
    img_edge_dir[img_edge_dir <= dir_low[0]] += 2*np.pi

    for i in range(4):
        # genrate the edge map
        edge_maps[:,:,i] = np.where(((img_edge_mag >= threshold) & (img_edge_dir >= dir_low[i]) & (img_edge_dir <= dir_up[i])), 255, 0)


    return edge_maps




# verify with circle image
circle = plt.imread('circle.jpg')
edge_maps = make_edge_map(circle, dx, dy)
edge_maps_in_row = [edge_maps[:, :, m] for m in range(edge_maps.shape[2])]
all_in_row = np.concatenate((edge_maps_in_row), axis=1)
plt.imshow(np.concatenate((circle, all_in_row), axis=1))
plt.title('Circle and edge orientations')
# plt.imshow(np.concatenate(np.dsplit(edge_maps, edge_maps.shape[2]), axis=0))
plt.show()

# now try with original image
edge_maps = make_edge_map(img, dx, dy)
edge_maps_in_row = [edge_maps[:, :, m] for m in range(edge_maps.shape[2])]
all_in_row = np.concatenate((edge_maps_in_row), axis=1)
plt.imshow(np.concatenate((img, all_in_row), axis=1))
plt.title('Original image and edge orientations')
plt.show()
