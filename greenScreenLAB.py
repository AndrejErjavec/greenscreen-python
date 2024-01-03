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

def mask_alpha(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
  out = image.copy()
  out = cv2.cvtColor(out, cv2.COLOR_BGR2BGRA)
  out[:,:,3] = mask
  return out

# https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)
def normalize_image(image: np.ndarray, alpha:int, beta:int) -> np.ndarray:
  min_val, max_val = np.min(image), np.max(image)
  normalized = (image - min_val) * ((beta - alpha) / (max_val - min_val)) + alpha
  return normalized.astype(np.uint8)

def binary_threshold(image: np.ndarray, threshold: int, max_val: int = 255, inverse:bool = False) -> np.ndarray:
  if len(image.shape) != 2:
    raise Exception('Shape of image not supported')
  
  out = np.zeros((image.shape[0], image.shape[1]))
  if inverse: out[image >= threshold] = max_val
  else: out[image <= threshold] = max_val
  return out 

def invert_image(image):
  return (255 - image).astype(np.uint8)

"""
RGB color space is not ideal for color-based segmentation

LAB color space: (L*, A*, B*)
L* --> luminosity (0..100)
A* --> green to red (-128..127) (-a* = green | +a* = red)
B? --> blue to yellow (-128..127) (-b* = blue | +b* = yellow)
"""

def removeGreenBackground(img: np.ndarray) -> np.ndarray:
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
  # masked = cv2.cvtColor(masked, cv2.COLOR_LAB2BGR)

  return masked, mask

def removeGreenEdge(image, threshold, mask):
  # convert back to LAB
  masked_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
  # normalize A channel to use entire [0:255] range
  masked_lab_norm = normalize_image(masked_lab[:,:,1], 0, 255)

  # Threshold normalized grayscale image to segment black border representing green outline on original masked image
  border_mask = binary_threshold(masked_lab_norm, threshold, 255)

  border_mask2 = invert_image(border_mask)
  final_mask = cv2.bitwise_and(mask, border_mask2)

  result = image.copy()
  result = apply_mask(result, final_mask)
  result[final_mask == 0] = 0
  result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
  return result, border_mask, final_mask
