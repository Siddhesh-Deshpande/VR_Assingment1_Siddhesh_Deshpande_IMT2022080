import os
import cv2
import numpy as np

def detect_features(image):
    # Detect keypoints and descriptors using SIFT
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    keypoints = np.float32([kp.pt for kp in keypoints])
    return keypoints, descriptors

def match_features(kpA, kpB, descA, descB, ratio=0.75, reproj_thresh=4.0):
    # Match features using BFMatcher and Lowe's ratio test
    matcher = cv2.BFMatcher()
    raw_matches = matcher.knnMatch(descA, descB, 2)
    good_matches = [(m[0].trainIdx, m[0].queryIdx) for m in raw_matches 
                    if len(m) == 2 and m[0].distance < m[1].distance * ratio]
    
    if len(good_matches) > 4:
        # Find homography using RANSAC
        ptsA = np.float32([kpA[i] for (_, i) in good_matches])
        ptsB = np.float32([kpB[i] for (i, _) in good_matches])
        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reproj_thresh)
        return good_matches, H, status
    
    return None

def draw_matches(imageA, imageB, kpA, kpB, matches, status, max_lines=100):
    # Draw matching lines between images
    hA, wA = imageA.shape[:2]
    hB, wB = imageB.shape[:2]
    result = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    result[:hA, :wA] = imageA
    result[:hB, wA:] = imageB
    
    line_count = 0
    for ((train_idx, query_idx), s) in zip(matches, status):
        if s and line_count < max_lines:
            ptA = (int(kpA[query_idx][0]), int(kpA[query_idx][1]))
            ptB = (int(kpB[train_idx][0]) + wA, int(kpB[train_idx][1]))
            cv2.line(result, ptA, ptB, (0, 0, 255), 1)
            line_count += 1
    
    return result

def crop_image(image):
    # Remove black borders from the stitched image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        return image[y:y + h - 1, x:x + w - 1]
    
    return image

def resize_image(image, width):
    # Resize image while maintaining aspect ratio
    h, w = image.shape[:2]
    new_height = int((width / float(w)) * h)
    return cv2.resize(image, (width, new_height), interpolation=cv2.INTER_AREA)

def stitch_images(image1, image2, visualize=False):
    # Stitch two images using feature matching and homography
    kpA, descA = detect_features(image2)
    kpB, descB = detect_features(image1)
    
    match_result = match_features(kpA, kpB, descA, descB)
    if not match_result:
        print("Not enough keypoints matched.")
        return None
    
    matches, H, status = match_result
    stitched = cv2.warpPerspective(image2, H, (image2.shape[1] + image1.shape[1], image2.shape[0]))
    stitched[0:image1.shape[0], 0:image1.shape[1]] = image1
    stitched = crop_image(stitched)
    
    if visualize:
        match_viz = draw_matches(image2, image1, kpA, kpB, matches, status)
        return stitched, match_viz
    
    return stitched

def process_folder(input_folder, output_folder):
    # Process images in a folder and stitch them sequentially
    image_paths = sorted(
        [os.path.join(input_folder, f) for f in os.listdir(input_folder)],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
    
    if len(image_paths) < 2:
        print("Need at least two images to stitch.")
        return
    
    base_image = resize_image(cv2.imread(image_paths[0]), width=600)
    os.makedirs(output_folder, exist_ok=True)
    
    for i in range(1, len(image_paths)):
        next_image = resize_image(cv2.imread(image_paths[i]), width=600)
        result = stitch_images(base_image, next_image, visualize=True)
        if result is None:
            continue
        stitched, match_viz = result
        cv2.imwrite(os.path.join(output_folder, f"match{i}.jpg"), match_viz)
        base_image = stitched
    
    cv2.imwrite(os.path.join(output_folder, "panorama.jpg"), base_image)

if __name__ == "__main__":
    if not os.path.exists("output_part2"):
        os.mkdir("output_part2")
    process_folder("input_part2", "output_part2")