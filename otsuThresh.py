# https://muthu.co/otsus-method-for-image-thresholding-explained-and-implemented/

import numpy as np

def __intraclass_variance(image: np.ndarray, threshold: int) -> float:
  c1 = image[image >= threshold]
  c2 = image[image < threshold]
  w1 = len(c1) / (image.shape[0]*image.shape[1])
  w2 = len(c2) / (image.shape[0]*image.shape[1])

  var1 = np.var(c1)
  var2 = np.var(c2)

  return w1*var1 + w2*var2
   

def __calculate_thresh(image: np.ndarray) -> int:
  vars = [(thresh, __intraclass_variance(image, thresh)) for thresh in range(np.min(image)+1, np.max(image))]
  min_v = min(vars, key = lambda x: x[1])
  return min_v[0]


def otsu_threshold(image: np.ndarray, max_val: int, inverted: bool = False) -> [np.ndarray, int]:
  out = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
  thresh_value = __calculate_thresh(image)
  if inverted: 
    out[image >= thresh_value] = max_val
  else:
    out[image < thresh_value] = max_val
  return out, thresh_value