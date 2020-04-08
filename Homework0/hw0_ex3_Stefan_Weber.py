import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Using the image stopturnsigns.jpg from ILIAS, try to nd the threshold values tmin, tmax,
# such that a mask m contains only pixels of the stop sign. An example of such a boolean
# binary mask can be seen below.
img = mpimg.imread('stopturnsigns.jpg')
img = img.copy()

mask = np.all((img > [230, 33, 50]) & (img < [255, 65, 95]), axis=-1)
mask2 = ~mask

img[mask] = [0, 0, 0]
img[mask2] = [255, 255, 255]
plt.imshow(img)
plt.show()
