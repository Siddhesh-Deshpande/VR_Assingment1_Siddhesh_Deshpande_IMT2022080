import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "coins.jpg"  # Update this path if needed
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Find contours of the detected edges
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
output = image.copy()
cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

# Convert BGR image to RGB for visualization in matplotlib
output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

# Display the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(edges, cmap='gray')
plt.title("Edge Detection")

plt.subplot(1, 2, 2)
plt.imshow(output_rgb)
plt.title("Detected Coins")

plt.show()