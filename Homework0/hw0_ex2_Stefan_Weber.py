import numpy as np
import matplotlib.pyplot as plt

# Write a function create image that generates a color image with the height/width
# dimensions of n x m with uniform randomly distributed red and black pixels. Add a
# single green pixel at a random location.
def create_image(n,m):
    img = np.zeros([n,m,3], dtype=np.uint8)
    img[:,:,0] = np.random.randint(2, size=(n, m)) * 255
    img[np.random.randint(n),np.random.randint(n),:] = [0, 255, 0]
    return img

# Next, write a function find pixels that nds the indexes of pixels with the values
# pixel values in an image img.
def find_pixels(img, pixel_values):     	  
     mask = np.all(img == pixel_values, axis=-1)
     result = np.argwhere(mask==True)
     return result

# Using the image img, compute the euclidean distance of each red pixel from the green
# pixel without the use of any for loops.
def compute_distances(img):
    greenPixel = find_pixels(img, [0,255,0])
    redPixels = find_pixels(img, [255, 0, 0])
    diff = greenPixel-redPixels
    dist = np.sqrt(np.square(diff[:,0])+np.square(diff[:,1]))
    return dist

# Display the computed distance vector dist in a histogram (with 100 bins), compute
# the mean, standard deviation and the median. Display the values as a title above the
# plot.
def visualize_results(dist):
    plt.hist(dist, bins =100)
    plt.title('distance mean=%f, std=%f, median=%f' %(np.median(dist), np.std(dist), np.mean(dist)))
    plt.show()

img = create_image(100,50)
plt.imshow(img)
plt.show()
dist = compute_distances(img)
visualize_results(dist)

