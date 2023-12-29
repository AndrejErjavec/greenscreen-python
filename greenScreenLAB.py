# Idea for implementation taken from:
# https://stackoverflow.com/questions/51719472/remove-green-background-screen-from-image-using-opencv-python

import cv2
import numpy as np
from matplotlib import pyplot as plt
from otsuThresh import otsu_threshold
from gaussianBlur import gaussian_blur

def apply_mask(image: np.ndarray, mask: np.ndarray, blackBackground: bool = False) -> np.ndarray:
  out = image.copy()
  if blackBackground: out[mask == 0] = 0
  else: out[mask == 0] = 255
  return out

# https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)
def normalizeImage(image: np.ndarray, alpha:int, beta:int):
  min_val, max_val = np.min(image), np.max(image)
  normalized = (image - min_val) * ((beta - alpha) / (max_val - min_val)) + alpha
  return normalized.astype(np.uint8)

def binary_threshold(image, threshold, max_val: int = 255, inverse:bool = False):
  if len(image.shape) != 2:
    raise Exception('Shape of image not supported')
  
  out = np.zeros((image.shape[0], image.shape[1]))
  if inverse: out[image >= threshold] = max_val
  else: out[image <= threshold] = max_val
  return out 

# load image
img = cv2.imread('images/greenscreen.jpg')

"""
RGB color space is not ideal for color-based segmentation

LAB color space: (L*, A*, B*)
L* --> luminosity (0..100)
A* --> green to red (-128..127) (-a* = green | +a* = red)
B? --> blue to yellow (-128..127) (-b* = blue | +b* = yellow)
"""

# convert to LAB
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
A = lab[:, :, 1]

# histogram of A* channel
#hist = cv2.calcHist([lab], [1], None, [256], [0, 256])
#plt.plot(hist)
#plt.show()

# create a "binary" mask [0,255]
[mask, thresh] = otsu_threshold(A, 255, inverted=True)

# apply mask to image
masked = apply_mask(img, mask)
cv2.imshow('masked', masked)
cv2.imwrite('result-before.png', masked)

# convert back to LAB
mlab = cv2.cvtColor(masked, cv2.COLOR_BGR2LAB)
# normalize A channel to use entire [0:255] range
dst = normalizeImage(mlab[:,:,1], 0, 255)
# Threshold normalized grayscale image to segment black border representing green outline on original masked image
[bt, btv] = otsu_threshold(dst, 255)
threshold_value = 70
dst_th = binary_threshold(dst, btv, 255)
cv2.imshow('dst th', dst_th)

# 127 represents white color in A channel (middle between green and red)
mlab2 = mlab.copy()
mlab[:,:,1][dst_th == 255] = 127
result = cv2.cvtColor(mlab, cv2.COLOR_LAB2BGR)

img2 = cv2.cvtColor(mlab, cv2.COLOR_LAB2BGR)
img2[mask==0]=(255,255,255)
cv2.imshow('result-after', result)
cv2.imwrite('result-after.png', result)

cv2.waitKey(0)
cv2.destroyAllWindows()