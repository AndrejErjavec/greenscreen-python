# https://mattmaulion.medium.com/chromaticity-segmentation-image-processing-cd168f4cf437
# https://medium.com/swlh/introduction-to-image-processing-part-5-image-segmentation-1-99f93d9f7a5e

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('images/greenscreen.jpg')
img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

histogram = cv2.calcHist([img], [0], None, [256], [0,256])

for i, col in enumerate(['b', 'g', 'r']):
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color = col)
    plt.xlim([0, 256])
    
plt.show()

# lower_green = np.array([52, 0, 55])
# upper_green = np.array([104, 255, 255])

# mask = cv2.inRange(img_hsv, lower_green, upper_green)
# mask = np.bitwise_not(mask)
# result = cv2.bitwise_and(img, img, mask=mask)

# cv2.imwrite('catHSVmasked.png', result)
# cv2.imshow('result', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()