import cv2
import numpy as np
import os

def preprocessing(image_name):
    """
    Preprocess the input image by converting it to grayscale, resizing it, 
    applying Gaussian blur, and performing adaptive thresholding.
    """
    image = cv2.imread(image_name)  # Load the image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Scale the image to a fixed size (700 pixels for the longest dimension)
    scale = 700 / max(image.shape[:2])
    image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    gray_image = cv2.resize(gray_image, (0, 0), fx=scale, fy=scale)

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply adaptive thresholding to obtain a binary image
    final_image = cv2.adaptiveThreshold(
        blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )

    return image, final_image, scale  # Return color image, processed image, and scale

def check_circularity(contour, scale):
    """
    Check if a given contour represents a circular shape.
    Uses the circularity formula: Circularity = 4π * (Area / Perimeter²).
    A circularity value close to 1 indicates a perfect circle.
    """
    perimeter = cv2.arcLength(contour, True)  # Calculate contour perimeter
    area = cv2.contourArea(contour)  # Calculate contour area

    # Avoid division by zero and check circularity constraints
    if perimeter:
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        
        # Consider contours with circularity near 1 and area greater than a threshold
        if 0.7 < circularity < 1.2 and area > 500 * (scale ** 2):
            return True
    
    return False

def coin_detection(image, scale):
    """
    Detects circular objects (coins) in the processed binary image.
    Filters contours based on circularity and size.
    """
    # Find external contours in the thresholded image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # List to store valid coin contours
    shapes = [cnt for cnt in contours if check_circularity(cnt, scale)]

    return shapes

def segmentation(original_image, contours):
    """
    Segments detected coins from the original image (natural color) and saves each coin separately.
    Also, draws red contours around detected coins on the output image.
    """
    output_image = original_image.copy()
    cv2.drawContours(output_image, contours, -1, (0, 0, 255), 2)  # Draw contours in red

    for idx, contour in enumerate(contours):
        # Get bounding box coordinates
        x, y, w, h = cv2.boundingRect(contour)

        # Create a blank mask (single-channel)
        mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Extract the coin region from the original colored image
        isolated_coin = cv2.bitwise_and(original_image, original_image, mask=mask)

        # Crop the coin from the extracted region
        cropped_coin = isolated_coin[y:y + h, x:x + w]

        # Save the segmented coin
        output_path = os.path.join("output_part1", f"segmented_coin_{idx}.png")
        cv2.imwrite(output_path, cropped_coin)

    return output_image  

if __name__ == "__main__":
    if not os.path.exists("output_part1"):
        os.mkdir("output_part1")


    original_image, processed_image, scale = preprocessing("coins1.jpg")
    coin_shapes = coin_detection(processed_image, scale)

    outlined_image = segmentation(original_image, coin_shapes)  


    filename = os.path.join("output_part1", "outlined.png")
    cv2.imwrite(filename, outlined_image)
    
    print(f"The number of coins in the image are {len(coin_shapes)}")
