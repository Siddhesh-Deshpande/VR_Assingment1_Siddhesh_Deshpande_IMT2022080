import cv2
import numpy as np

# Load the image
image = cv2.imread('coins.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to remove noise
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# Use adaptive thresholding
binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY_INV, 11, 3)

# Perform morphological closing to fill small holes
kernel = np.ones((5, 5), np.uint8)
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

# Find contours
contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on size
min_contour_area = 500  # Adjust based on image resolution
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

# Count coins
num_coins = len(filtered_contours)
print(f"Total Coins Detected: {num_coins}")

# Draw contours
output_image = image.copy()
cv2.drawContours(output_image, filtered_contours, -1, (0, 255, 0), 2)

# Show results
cv2.imshow("Original Image", image)
cv2.imshow("Binary Image", binary)
cv2.imshow("Filtered Contours", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# getting 11 coins in the output.