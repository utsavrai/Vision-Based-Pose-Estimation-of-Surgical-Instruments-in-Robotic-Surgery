# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Load stereo images
# left_image_path = '/home/utsav/IProject/data/3d_adjust/rectified_left.png'
# right_image_path = '/home/utsav/IProject/data/3d_adjust/rectified_right.png'

# left_image = cv2.imread(left_image_path, cv2.IMREAD_COLOR)
# right_image = cv2.imread(right_image_path, cv2.IMREAD_COLOR)

# if left_image is None or right_image is None:
#     raise FileNotFoundError("One or both of the stereo images couldn't be loaded. Check the file paths.")

# # Load camera calibration parameters (these should be pre-calculated)
# # For the purpose of this example, we'll assume these are given
# # Replace these with actual calibration data
# K1 = np.array([[847.28300117, 0, 537.40095139], [0, 847.28300117, 309.78999329], [0, 0, 1]])
# K2 = np.array([[847.28300117, 0, 537.40095139], [0, 847.28300117, 309.78999329], [0, 0, 1]])
# D1 = np.zeros(5)  # Distortion coefficients for left camera
# D2 = np.zeros(5)  # Distortion coefficients for right camera
# R = np.eye(3)  # Rotation matrix between cameras
# T = np.array([5.5160, 0.0481, -0.0733])  # Translation vector between cameras

# # Step 1: Visualize epipolar lines (pre-rectification)
# def draw_epipolar_lines(img1, img2, lines, pts1, pts2):
#     ''' img1 - image on which we draw the epilines for the points in img2
#         lines - corresponding epilines '''
#     r, c = img1.shape[:2]
#     if len(img1.shape) == 2 or img1.shape[2] == 1:
#         img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
#     if len(img2.shape) == 2 or img2.shape[2] == 1:
#         img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
#     for r, pt1, pt2 in zip(lines, pts1, pts2):
#         color = tuple(np.random.randint(0, 255, 3).tolist())
#         x0, y0 = map(int, [0, -r[2] / r[1]])
#         x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
#         img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
#         img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
#         img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
#     return img1, img2

# # Find the fundamental matrix
# sift = cv2.SIFT_create()
# kp1, des1 = sift.detectAndCompute(left_image, None)
# kp2, des2 = sift.detectAndCompute(right_image, None)
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1, des2, k=2)

# # Apply ratio test
# good_matches = []
# pts1 = []
# pts2 = []
# for m, n in matches:
#     if m.distance < 0.75 * n.distance:
#         good_matches.append(m)
#         pts1.append(kp1[m.queryIdx].pt)
#         pts2.append(kp2[m.trainIdx].pt)

# pts1 = np.int32(pts1)
# pts2 = np.int32(pts2)
# F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

# # Select inlier points
# pts1 = pts1[mask.ravel() == 1]
# pts2 = pts2[mask.ravel() == 1]

# # Find epilines corresponding to points in right image and draw them on the left image
# lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
# lines1 = lines1.reshape(-1, 3)
# img5, img6 = draw_epipolar_lines(left_image, right_image, lines1, pts1, pts2)

# # Find epilines corresponding to points in left image and draw them on the right image
# lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
# lines2 = lines2.reshape(-1, 3)
# img3, img4 = draw_epipolar_lines(right_image, left_image, lines2, pts2, pts1)

# plt.figure(figsize=(12, 6))
# plt.subplot(121), plt.imshow(img5)
# plt.subplot(122), plt.imshow(img3)
# plt.suptitle('Epipolar Lines (before rectification)')
# plt.show()

# # Step 2: Stereo rectification
# h, w = left_image.shape[:2]
# R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, D1, K2, D2, (w, h), R, T)

# # Compute the rectification map
# left_map1, left_map2 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w, h), cv2.CV_16SC2)
# right_map1, right_map2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (w, h), cv2.CV_16SC2)

# left_rectified = cv2.remap(left_image, left_map1, left_map2, cv2.INTER_LINEAR)
# right_rectified = cv2.remap(right_image, right_map1, right_map2, cv2.INTER_LINEAR)

# # Visualize rectified images with epipolar lines
# def draw_lines(img1, img2, n_lines=10):
#     ''' Draw horizontal lines across the pair of images '''
#     img1 = img1.copy()
#     img2 = img2.copy()
#     h, w = img1.shape[:2]
#     line_spacing = h // n_lines
#     for y in range(0, h, line_spacing):
#         img1 = cv2.line(img1, (0, y), (w, y), (0, 255, 0), 1)
#         img2 = cv2.line(img2, (0, y), (w, y), (0, 255, 0), 1)
#     return img1, img2

# left_rectified_lines, right_rectified_lines = draw_lines(left_rectified, right_rectified)

# plt.figure(figsize=(12, 6))
# plt.subplot(121), plt.imshow(left_rectified_lines)
# plt.subplot(122), plt.imshow(right_rectified_lines)
# plt.suptitle('Rectified Images with Epipolar Lines')
# plt.show()



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

def main():

    idx = 899
    # idx = 210

    # left_image_path = "/home/utsav/IProject/data/captured/lnd1/rect_left/{}.png".format(idx)
    # right_image_path = "/home/utsav/IProject/data/captured/lnd1/rect_right/{}.png".format(idx)

    
    left_image_path = '/home/utsav/IProject/data/captured/rectified_left_image.png'  # Update with actual path
    right_image_path = '/home/utsav/IProject/data/captured/rectified_right_image.png'  # Update with actual path
    # left_image_path = '/home/utsav/IProject/data/3d_adjust/rectified_left.png'  # Update with actual path
    # right_image_path = '/home/utsav/IProject/data/3d_adjust/rectified_right.png'  # Update with actual path

    left_img, right_img = load_images(left_image_path, right_image_path)
    kp1, kp2, matches = detect_and_match_features(left_img, right_img)
    matched_img = draw_epipolar_lines(left_img, right_img, kp1, kp2, matches)

    # Display the images with epipolar lines
    plt.figure(figsize=(20, 10))
    plt.imshow(matched_img)
    plt.title('Epipolar Lines on Matched Features')
    plt.show()

if __name__ == "__main__":
    main()
