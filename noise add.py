import cv2
import numpy as np

# original image
f = cv2.imread('dgu_gray.png', 0)
f = f/255 

x, y = f.shape
mean = 0
var = 0.01
sigma = np.sqrt(var)
n = np.random.normal(loc=mean, 
                     scale=sigma, 
                     size=(x,y))

# add a gaussian noise
g = f + n




cv2.imshow('original image', f)
cv2.imshow('Gaussian noise', n)
cv2.imshow('Corrupted Image', g)

#cv2.imshow('Gaussian Filter', img1D.astype(np.uint8))

cv2.waitKey(0)
cv2.destroyAllWindows()