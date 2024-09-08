import cv2
import numpy as np
import os
import glob
import yaml

def load_yaml_data(path):
    with open(path) as f_tmp:
        return yaml.load(f_tmp, Loader=yaml.FullLoader)
        # return yaml.load(f_tmp)

def load_cam_params_dict(file_path):
    dual_cam_params = load_yaml_data(file_path)

    R = np.array(dual_cam_params['cam']['R']['data']).reshape((3,3))
    T = np.array(dual_cam_params['cam']['T']['data'])
    camera1_matrix = np.array(dual_cam_params['cam']['M1']['data']).reshape((3,3))
    camera2_matrix = np.array(dual_cam_params['cam']['M2']['data']).reshape((3,3))
    camera1_distortion = np.array(dual_cam_params['cam']['D1']['data'])
    camera2_distortion = np.array(dual_cam_params['cam']['D2']['data'])

    return R,T,camera1_matrix, camera1_distortion,camera2_matrix, camera2_distortion

def rectify(m1, d1, m2, d2, width, height, r, t):

    R1, R2, P1, P2, Q, _roi1, _roi2 = \
            cv2.stereoRectify(cameraMatrix1=m1,
                            distCoeffs1=d1,
                            cameraMatrix2=m2,
                            distCoeffs2=d2,
                            imageSize=(width, height),
                            R=r,
                            T=t,
                            flags=0,
                            alpha=0.0
                            )
    '''
    R1/R2: Output 3x3 rectification transform (rotation matrix) for the first/second camera
    P1/P2: Output 3x4 projection matrix in the new (rectified) coordinate systems for the first/second camera
    Q: Output 4 disparity-to-depth mapping matrix
    '''
    map1_x, map1_y = cv2.initUndistortRectifyMap(
        cameraMatrix=m1,
        distCoeffs=d1,
        R=R1,
        newCameraMatrix=P1,
        size=(width, height),
        m1type=cv2.CV_32FC1)

    map2_x, map2_y = cv2.initUndistortRectifyMap(
        cameraMatrix=m2,
        distCoeffs=d2,
        R=R2,
        newCameraMatrix=P2,
        size=(width, height),
        m1type=cv2.CV_32FC1)
    print('Q',Q)
    print('P1',P1)
    print('P2',P2)

    f = Q[2, 3]
    baseline = 1./Q[3, 2]

    return map1_x, map1_y, map2_x, map2_y, f, baseline, Q, P1,P2


class StereoDepthVisualizer:
    def __init__(self, left_img_path, right_img_path):
        self.left_img_og = cv2.imread(left_img_path)
        self.right_img_og = cv2.imread(right_img_path)
        self.left_points = []
        self.right_points = []
        self.current_image = 'left'


        file_path = '/home/utsav/IProject/data/captured/lnd1/matlabconfig.yaml'
        self.R,self.T,m1, d1, m2, d2 = load_cam_params_dict(file_path)

        IMG_HEIGHT,IMG_WIDTH = 540, 960
        self.map1_x, self.map1_y, self.map2_x, self.map2_y, self.focal_length, self.baseline, self.Q, self.P1, self.P2 = \
            rectify(m1, d1, m2, d2, IMG_WIDTH, IMG_HEIGHT, self.R, self.T)
        
        self.left_img = cv2.remap(self.left_img_og, self.map1_x, self.map1_y, cv2.INTER_LINEAR)
        self.right_img = cv2.remap(self.right_img_og, self.map2_x, self.map2_y, cv2.INTER_LINEAR)
        # Save the rectified images
        cv2.imwrite('rectified_left.png', self.left_img)
        cv2.imwrite('rectified_right.png', self.right_img)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_image == 'left':
                self.left_points.append((x, y))
                cv2.circle(self.left_img, (x, y), 5, (0, 255, 0), -1)
                if len(self.left_points) == 2:
                    self.current_image = 'right'
                    print("Now click on the corresponding points in the right image.")
            elif self.current_image == 'right':
                self.right_points.append((x, y))
                cv2.circle(self.right_img, (x, y), 5, (0, 255, 0), -1)
                if len(self.right_points) == 2:
                    self.calculate_and_draw()

    def calculate_and_draw(self):
        # Calculate disparities
        disparities = [abs(lp[0] - rp[0]) for lp, rp in zip(self.left_points, self.right_points)]
        
        # Calculate depths
        depths = [(self.focal_length * self.baseline) / (d + self.P2[0][2] - self.P1[0][2]) for d in disparities]
        
        # Calculate 3D coordinates
        points_3d = []
        for i in range(2):
            z = depths[i]
            x = (self.left_points[i][0] - self.left_img.shape[1]/2) * z / self.focal_length
            y = (self.left_points[i][1] - self.left_img.shape[0]/2) * z / self.focal_length
            points_3d.append((x, y, z))
        # Calculate distance
        distance = np.sqrt(sum((a-b)**2 for a, b in zip(points_3d[0], points_3d[1])))
        
        # Draw line and display distance on left image
        cv2.line(self.left_img, self.left_points[0], self.left_points[1], (0, 0, 255), 2)
        mid_point = ((self.left_points[0][0] + self.left_points[1][0]) // 2, 
                     (self.left_points[0][1] + self.left_points[1][1]) // 2)
        cv2.putText(self.left_img, f"{distance:.2f}", mid_point, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        print(f"Distance between points: {distance:.2f} units")
        print("You can select two new points in the left image.")
        self.current_image = 'left'
        self.left_points = []
        self.right_points = []

    def run(self):
        cv2.namedWindow('Left Image')
        cv2.namedWindow('Right Image')
        cv2.setMouseCallback('Left Image', self.mouse_callback)
        cv2.setMouseCallback('Right Image', self.mouse_callback)

        print("Click on two points in the left image.")

        while True:
            cv2.imshow('Left Image', self.left_img)
            cv2.imshow('Right Image', self.right_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    idx = 1252
    # idx = 210

    left_img_path = "/home/utsav/IProject/data/captured/lnd1/left/{}.png".format(idx)
    right_img_path = "/home/utsav/IProject/data/captured/lnd1/right/{}.png".format(idx)
    # focal_length = 801.57991404  # camera's focal length in pixels
    # baseline = 5.5160  # stereo camera baseline in meters

    verifier = StereoDepthVisualizer(left_img_path, right_img_path)
    verifier.run()