import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_images(left_image_path, right_image_path):
    left_img = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)
    if left_img is None or right_img is None:
        raise FileNotFoundError("One or both of the images couldn't be loaded. Check the file paths.")
    return left_img, right_img

def detect_and_match_features(left_img, right_img):
    # Use ORB detector to find keypoints and descriptors
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(left_img, None)
    kp2, des2 = orb.detectAndCompute(right_img, None)

    # Use BFMatcher to find matches between descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Sort matches based on distance
    matches = sorted(matches, key=lambda x: x.distance)
    return kp1, kp2, matches

def draw_epipolar_lines(left_img, right_img, kp1, kp2, matches, num_matches_to_draw=50):
    # Draw the matches on the images
    matched_img = cv2.drawMatches(left_img, kp1, right_img, kp2, matches[:num_matches_to_draw], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Draw epipolar lines (horizontal lines on the images)
    h1, w1 = left_img.shape
    h2, w2 = right_img.shape
    for i in range(num_matches_to_draw):
        pt1 = np.int32(kp1[matches[i].queryIdx].pt)
        pt2 = np.int32(kp2[matches[i].trainIdx].pt)
        
        # Epipolar lines for left image
        cv2.line(matched_img, (0, pt1[1]), (w1, pt1[1]), (0, 255, 0), 1)
        
        # Epipolar lines for right image
        cv2.line(matched_img, (w1, pt2[1]), (w1 + w2, pt2[1]), (0, 255, 0), 1)

    return matched_img

def compare_descriptors(des1, des2):
    if des1.shape != des2.shape:
        return False
    return np.array_equal(des1, des2)

def main():
    # Paths to images
    idx = 899

    left_image_path_1 = "/home/utsav/IProject/data/captured/lnd1/rect_left/{}.png".format(idx)
    right_image_path_1 = "/home/utsav/IProject/data/captured/lnd1/rect_right/{}.png".format(idx)
    # left_image_path_2 = "/home/utsav/IProject/data/lnd1_old/rect_left/{}.png".format(idx)
    # right_image_path_2 = "/home/utsav/IProject/data/lnd1_old/rect_right/{}.png".format(idx)

    left_image_path_2 = '/home/utsav/IProject/data/l899.png'  # Update with actual path
    right_image_path_2 = '/home/utsav/IProject/data/r899.png'  # Update with actual path


    # Load images
    left_img_1, right_img_1 = load_images(left_image_path_1, right_image_path_1)
    left_img_2, right_img_2 = load_images(left_image_path_2, right_image_path_2)
    
    # Detect and match features for both pairs
    kp1_1, kp2_1, matches_1 = detect_and_match_features(left_img_1, right_img_1)
    kp1_2, kp2_2, matches_2 = detect_and_match_features(left_img_2, right_img_2)
    
    # Draw epipolar lines for both pairs
    matched_img_1 = draw_epipolar_lines(left_img_1, right_img_1, kp1_1, kp2_1, matches_1)
    matched_img_2 = draw_epipolar_lines(left_img_2, right_img_2, kp1_2, kp2_2, matches_2)

    # Display the images with epipolar lines
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(matched_img_1)
    plt.title('Epipolar Lines on Matched Features (Rectified Pair 1)')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(matched_img_2)
    plt.title('Epipolar Lines on Matched Features (Rectified Pair 2)')
    plt.axis('off')
    plt.show()

    # Compare descriptors
    des1_1 = [kp1_1[match.queryIdx].pt for match in matches_1]
    des2_1 = [kp2_1[match.trainIdx].pt for match in matches_1]
    des1_2 = [kp1_2[match.queryIdx].pt for match in matches_2]
    des2_2 = [kp2_2[match.trainIdx].pt for match in matches_2]

    same_descriptors = compare_descriptors(np.array(des1_1), np.array(des1_2)) and compare_descriptors(np.array(des2_1), np.array(des2_2))
    if same_descriptors:
        print("Both pairs of rectified images have the same descriptors.")
    else:
        print("The descriptors differ between the two pairs of rectified images.")

if __name__ == "__main__":
    main()
