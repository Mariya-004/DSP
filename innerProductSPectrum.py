from google.colab import files
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from matplotlib import image as mpimg

# Upload images
uploaded = files.upload()

# Assuming you uploaded 'test.png' for both images
image_path1 = 'test.png'  # Change to the actual name if necessary
image_path2 = 'test.png'  # Change to the actual name if necessary

# Load the images
img1 = mpimg.imread(image_path1)  # First image
img2 = mpimg.imread(image_path2)  # Second image

# If images have multiple channels (e.g., RGB), convert them to grayscale
if img1.ndim == 3:
    img1 = np.mean(img1, axis=-1)  # Convert to grayscale by averaging the channels
if img2.ndim == 3:
    img2 = np.mean(img2, axis=-1)  # Convert to grayscale by averaging the channels

# Compute the 2D Fourier Transforms (Frequency domain)
F1 = fft2(img1)
F2 = fft2(img2)

# Shift the zero-frequency component to the center (for better visualization)
F1_shifted = fftshift(F1)
F2_shifted = fftshift(F2)

# Compute the inner product in the spatial domain
inner_product_spatial = np.sum(img1 * img2)

# Compute the inner product in the frequency domain (complex conjugate of F2)
inner_product_frequency = np.sum(F1_shifted * np.conj(F2_shifted)) / img1.size

# Print the results
print(f"Inner product in the spatial domain: {inner_product_spatial}")
print(f"Inner product in the frequency domain: {inner_product_frequency}")

# Optionally, visualize the images and their frequency domain representations
plt.figure(figsize=(10, 5))

# Plot the first image
plt.subplot(2, 2, 1)
plt.imshow(img1, cmap='gray')
plt.title('First Image')

# Plot the second image
plt.subplot(2, 2, 2)
plt.imshow(img2, cmap='gray')
plt.title('Second Image')

# Visualize the frequency domain (log scale) of the first image
plt.subplot(2, 2, 3)
plt.imshow(np.log(np.abs(F1_shifted) + 1), cmap='gray')
plt.title('Frequency Domain (First Image)')

# Visualize the frequency domain (log scale) of the second image
plt.subplot(2, 2, 4)
plt.imshow(np.log(np.abs(F2_shifted) + 1), cmap='gray')
plt.title('Frequency Domain (Second Image)')

plt.tight_layout()
plt.show()

