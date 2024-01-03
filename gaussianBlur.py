import numpy as np

"""
sigma -->
radius --> kernel size (blur strength)
"""
def __gaussian_filter(sigma: int, radius: int = None) -> np.ndarray:
  if radius == None or radius == 0:
    radius = 2 * int(4 * sigma + 0.5) + 1

  # [h, w] = size
  # hh = h // 2
  # wh = w // 2
    
  # force odd kernel size to ensure there is always a center pixel in the kernel
  kernel_size = (2 * radius + 1, 2 * radius + 1)
  gaussian_filter = np.zeros(kernel_size, np.float32)

  for i in range(-radius, radius):
    for j in range(-radius, radius):
      base = 1 / (2.0 * np.pi * sigma**2)
      exp = np.exp(-(i**2 + j**2) / (2.0 * sigma**2))
      G_sigma = base * exp
      gaussian_filter[i+radius, j+radius] = G_sigma
  
  # normalize the kernel (values add up to 1)
  gaussian_filter = gaussian_filter / np.sum(gaussian_filter)
  return gaussian_filter


def __convolution2D(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
  if len(image.shape) != 2:
    raise Exception('Shape of image not supported')

  [image_h, image_w] = image.shape
  [kernel_h, kernel_w] = kernel.shape

  radius_x = kernel_h // 2 # horizontal radius
  radius_y = kernel_w // 2 # vertical radius

  # pad the image
  padded = np.zeros((image_h + 2*radius_y, image_w + 2*radius_x))
  padded[radius_y:image_h+radius_y, radius_x:image_w + radius_x] = image.copy()

  [padded_h, padded_w] = padded.shape

  # fill edge (mirror fill) - this is equal to borderType cv2.BORDER_DEFAULT
  # top edge
  padded[0:radius_y, radius_x:radius_x+image_w] = np.flipud(image[0:radius_y, :])

  # bottom edge
  padded[radius_y+image_h:2*radius_y+image_h, radius_x : radius_x+image_w] = np.flipud(image[image_h-radius_y:image_h, :])

  # left edge
  padded[radius_y:padded_h - radius_y, 0:radius_x] = np.fliplr(image[:, 0:radius_x])

  # right edge
  padded[radius_y:padded_h-radius_y, image_w+radius_x:padded_w] = np.fliplr(image[:, image_w-radius_x:image_w])

  # corners
  
  out = np.zeros(image.shape, dtype=np.float32)

  # (i,j) inndicates the position of kernel center relative to the padded image
  for i in range(radius_y, radius_y + image_h):
    for j in range(radius_x, radius_x + image_w):
      sub_matrix = padded[i-radius_x:i+radius_x+1, j-radius_y:j+radius_y+1]
      out[i-radius_y, j-radius_x] = np.sum(sub_matrix * kernel)

  return out


def __convolution3D(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
  if len(image.shape) != 3:
    raise Exception('Shape of image not supported')

  [image_h, image_w, image_d] = image.shape
  [kernel_h, kernel_w] = kernel.shape

  radius_x = kernel_h // 2 # horizontal radius
  radius_y = kernel_w // 2 # vertical radius

  # pad the image
  padded = np.zeros((image_h + 2*radius_y, image_w + 2*radius_x, image_d))
  padded[radius_y:image_h+radius_y, radius_x:image_w + radius_x, :] = image.copy()

  [padded_h, padded_w, padded_d] = padded.shape

  # fill edge (mirror fill) - this is equal to borderType cv2.BORDER_DEFAULT
  # top edge
  padded[0:radius_y, radius_x:radius_x+image_w] = np.flipud(image[0:radius_y, :])

  # bottom edge
  padded[radius_y+image_h:2*radius_y+image_h, radius_x : radius_x+image_w] = np.flipud(image[image_h-radius_y:image_h, :])

  # left edge
  padded[radius_y:padded_h - radius_y, 0:radius_x] = np.fliplr(image[:, 0:radius_x])

  # right edge
  padded[radius_y:padded_h-radius_y, image_w+radius_x:padded_w] = np.fliplr(image[:, image_w-radius_x:image_w])

  # corners
  
  out = np.zeros(image.shape, dtype=np.float32)
  
  # (i,j) inndicates the position of kernel center relative to the padded image
  for i in range(radius_y, radius_y + image_h):
    for j in range(radius_x, radius_x + image_w):
      for c in range(image_d):
        sub_matrix = padded[i-radius_x:i+radius_x+1, j-radius_y:j+radius_y+1, c]
        out[i-radius_y, j-radius_x, c] = np.sum(sub_matrix * kernel)

  return out


def gaussian_blur(image: np.ndarray, sigma: int, radius: int = None) -> np.ndarray:
  kernel = __gaussian_filter(sigma, radius)

  if (len(image.shape)) == 3:
    blurred = __convolution3D(image, kernel)
  else:
    blurred = __convolution2D(image, kernel)

  return blurred.astype(np.uint8)
