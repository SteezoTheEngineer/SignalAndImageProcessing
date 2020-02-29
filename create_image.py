import numpy as np
import matplotlib.pyplot as plt

def create_image(n,m):
    img = np.zeros([n,m,3], dtype=np.uint8)
    img[:,:,0] = np.random.randint(2, size=(n, m)) * 255
    img[np.random.randint(n),np.random.randint(n),:] = [0, 255, 0]
    return img

def find_pixels(img, pixel_values):     	  
     mask = np.all(img == pixel_values, axis=-1)
     result = np.argwhere(mask==True)
     return result

def compute_distances(img):
    greenPixel = find_pixels(img, [0,255,0])
    redPixels = find_pixels(img, [255, 0, 0])
    diff = greenPixel-redPixels
    dist = np.sqrt(np.square(diff))
    return dist

def visualize_results(dist):
    plt.hist(dist, bins =100)
    plt.title('distance mean=%f, std=%f, median=%f' %(np.median(dist), np.std(dist), np.mean(dist)))
    plt.show()

img = create_image(3,3)
find_pixels(img, [0,255,0])
dist = compute_distances(img)
visualize_results(dist)

