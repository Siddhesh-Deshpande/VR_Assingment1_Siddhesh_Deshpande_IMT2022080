import cv2
import numpy as np

def detect_keypoints_and_match(img1, img2):
    """Detects keypoints and matches them using SIFT and FLANN-based matcher."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Use SIFT for feature detection
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # FLANN-based matcher (fast for SIFT)
    index_params = dict(algorithm=1, trees=5)  # KD-Tree
    search_params = dict(checks=50)  
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Apply Lowe's ratio test for better matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:  # Lowe's ratio test
            good_matches.append(m)

    return keypoints1, keypoints2, good_matches

def stitch_images(img1, img2):
    """Stitches two images using keypoint matching and homography."""
    keypoints1, keypoints2, good_matches = detect_keypoints_and_match(img1, img2)

    if len(good_matches) < 4:  # Homography needs at least 4 matches
        print("Not enough good matches!")
        return None

    # Extract coordinates of matched points
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

    # Compute Homography matrix
    H, _ = cv2.findHomography(points2, points1, cv2.RANSAC)

    # Warp img2 to img1's perspective
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]
    img2_warped = cv2.warpPerspective(img2, H, (width1 + width2, height1))

    # Place img1 on the left side
    img2_warped[0:height1, 0:width1] = img1

    # Convert to grayscale to find non-black areas
    gray = cv2.cvtColor(img2_warped, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find bounding box of the non-black area
    coords = cv2.findNonZero(thresh)
    x, y, w, h = cv2.boundingRect(coords)

    # Crop the valid region
    stitched = img2_warped[y:y+h, x:x+w]

    return stitched

# Load overlapping images
img1 = cv2.imread("image1.jpeg")  # Left image
img2 = cv2.imread("image2.jpeg")  # Right image

# Perform stitching
panorama = stitch_images(img1, img2)

if panorama is not None:
    # Resize for display (Optional)
    scale_percent = 50  # Adjust this to control output size
    width = int(panorama.shape[1] * scale_percent / 100)
    height = int(panorama.shape[0] * scale_percent / 100)
    resized_panorama = cv2.resize(panorama, (width, height))

    # Show and save the final stitched image
    cv2.imshow("Panorama", resized_panorama)
    cv2.imwrite("stitched_panorama.jpg", panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Failed to create panorama.")
