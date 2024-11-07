import imageio
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

image = imageio.imread('penguin.jpg', mode='L')

hist, bins = np.histogram(image.flatten(), bins=256, range=[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()

cdf_m = np.ma.masked_equal(cdf, 0)
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
cdf = np.ma.filled(cdf_m, 0).astype('uint8')

image_equalized = cdf[image]

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Equalized Image')
plt.imshow(image_equalized, cmap='gray')
plt.axis('off')

plt.show()
