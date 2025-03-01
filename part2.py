import os
import cv2
import numpy as np

def detect_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    keypoints = np.float32([kp.pt for kp in keypoints])
    return keypoints, descriptors

def match_features(kp1, kp2, desc1, desc2, ratio=0.75, reproj_thresh=4.0):
    matcher = cv2.BFMatcher()
    raw_matches = matcher.knnMatch(desc1, desc2, 2)
    good_matches = [(m[0].trainIdx, m[0].queryIdx) for m in raw_matches if len(m) == 2 and m[0].distance < m[1].distance * ratio]
    
    if len(good_matches) > 4:
        pts1 = np.float32([kp1[i] for (_, i) in good_matches])
        pts2 = np.float32([kp2[i] for (i, _) in good_matches])
        H, status = cv2.findHomography(pts1, pts2, cv2.RANSAC, reproj_thresh)
        return good_matches, H, status
    
    return None

def draw_matches(image1, image2, kp1, kp2, matches, status, max_lines=100):
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    result = np.zeros((max(h1, h2), w1 + w2, 3), dtype="uint8")
    result[:h1, :w1] = image1
    result[:h2, w1:] = image2
    
    for i, ((train_idx, query_idx), s) in enumerate(zip(matches, status)):
        if s and i < max_lines:
            pt1 = (int(kp1[query_idx][0]), int(kp1[query_idx][1]))
            pt2 = (int(kp2[train_idx][0]) + w1, int(kp2[train_idx][1]))
            cv2.line(result, pt1, pt2, (0, 0, 255), 1)
    
    return result

def crop_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        return image[y:y + h, x:x + w]
    
    return image

def resize_image(image, width):
    h, w = image.shape[:2]
    new_height = int((width / float(w)) * h)
    return cv2.resize(image, (width, new_height), interpolation=cv2.INTER_AREA)

def stitch_images(image1, image2, visualize=False):
    kp1, desc1 = detect_features(image1)
    kp2, desc2 = detect_features(image2)
    
    match_result = match_features(kp1, kp2, desc1, desc2)
    if not match_result:
        print("Not enough keypoints matched.")
        return None
    
    matches, H, status = match_result
    stitched = cv2.warpPerspective(image1, H, (image1.shape[1] + image2.shape[1], image1.shape[0]))
    stitched[0:image2.shape[0], 0:image2.shape[1]] = image2
    stitched = crop_image(stitched)
    
    if visualize:
        match_viz = draw_matches(image1, image2, kp1, kp2, matches, status)
        return stitched, match_viz
    
    return stitched

def process_folder(input_folder, output_folder):
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
        
        base_image, match_viz = result
        cv2.imwrite(os.path.join(output_folder, f"matches_{i}.jpg"), match_viz)
    
    cv2.imwrite(os.path.join(output_folder, "panorama.jpg"), base_image)
    print(f"Stitching complete. Results saved in {output_folder}.")
    
if __name__ == "__main__":
    
    process_folder("input_part2", "output_part2")
