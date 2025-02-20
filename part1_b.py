import cv2
import numpy as np
import os 

# Load the image
image = cv2.imread("coins.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)

# Apply adaptive thresholding to get a binary image
binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY_INV, 11, 3)

# Perform Morphological operations to close small holes
kernel = np.ones((3, 3), np.uint8)
morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

# Find contours
contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
contour_image = image.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)  # Get bounding box
    coin = image[y:y+h, x:x+w]  # Crop the coin
    filename = os.path.join("images",f"coin_{i}.png")
    cv2.imwrite(filename, coin)  # Save each coin separately

# Show the result
cv2.imshow("Segmented Coins", contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

