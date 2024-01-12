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
  # result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
  return result, border_mask, final_mask


def alpha_composite(src, dst, src_opacity, mask=None):
    srcRGB = src[:, :, :3]
    dstRGB = dst[:, :, :3]

    mask_b = mask == 255

    srcAlpha = src[:, :, 3] / 255.0
    #src[:, :, 3] / 255.0 * src_opacity
    srcAlpha[mask_b] = src_opacity
    dstAlpha = dst[:, :, 3] / 255.0

    outAlpha = srcAlpha + dstAlpha * (1 - srcAlpha)

    outRGB = (srcRGB * srcAlpha[..., np.newaxis] + dstRGB * dstAlpha[..., np.newaxis] * (1 - srcAlpha[..., np.newaxis])) / outAlpha[..., np.newaxis]
    outRGBA = np.dstack((outRGB, outAlpha * 255)).astype(np.uint8)

    return outRGBA

def main():
  EDGE_THRESHOLD = 100
  IMAGE_PATH = 'images/greenscreen.jpg'
  BACKGROUND_PATH = 'images/background.jpg'

  x = 500
  y = 200

  image = cv2.imread(IMAGE_PATH)
  background = cv2.imread(BACKGROUND_PATH)

  removed_bg, subject_mask = removeGreenBackground(image)
  removed_edge, border_mask, full_mask = removeGreenEdge(removed_bg, EDGE_THRESHOLD, subject_mask)
  removed_bg = cv2.cvtColor(removed_bg, cv2.COLOR_BGR2LAB)
  removed_bg[:,:,1][border_mask == 255] = 127
  removed_bg = cv2.cvtColor(removed_bg, cv2.COLOR_LAB2BGR)
  removed_bg = mask_alpha(removed_bg, subject_mask)

  im_h, im_w, _ = image.shape
  background_cut = background[y:y+im_h, x:x+im_w]
  background_cut = cv2.cvtColor(background_cut, cv2.COLOR_BGR2BGRA)

  result = alpha_composite(removed_bg, background_cut, 0.5, border_mask)

  cv2.imshow('a', result)
  cv2.imwrite('result.png', result)

  """
  crop = removed_edge[20:removed_edge.shape[0]-20, 20:removed_edge.shape[1]-20]
  crop = cv2.resize(crop, (removed_edge.shape[1], removed_edge.shape[0]))
  crop_blur = gaussian_blur(crop, 2, 2)

  #removed_edge_gray = cv2.cvtColor(removed_edge, cv2.COLOR_BGR2GRAY)
  removed_edge_t = mask_alpha(removed_edge, full_mask)
  #removed_edge_rgba = cv2.cvtColor(removed_edge_t, cv2.COLOR_BGR2BGRA)
  combined = overlay_transparent(crop_blur, removed_edge_t, 0, 0)
  combined_masked = apply_mask(combined, subject_mask)

  cv2.imshow('removed edge', removed_edge)
  cv2.imshow('crop', crop)
  cv2.imshow('combined', combined)
  cv2.imshow('result', combined_masked)

  cv2.imwrite('result.png', combined_masked)
  """

  cv2.waitKey()
  cv2.destroyAllWindows()

if __name__=="__main__":
  main()