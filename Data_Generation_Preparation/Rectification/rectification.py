from numpy import genfromtxt
import os
import cv2
import numpy as np
import glob
import yaml
np.set_printoptions(suppress=True)

IMG_WIDTH = 960
IMG_HEIGHT = 540
# GET_DISP_MAPS = True
# GET_DEPTH_MAPS = False

def load_cam_params_dict(file_path):
    dual_cam_params = load_yaml_data(file_path)

    R = np.array(dual_cam_params['cam']['R']['data']).reshape((3,3))
    T = np.array(dual_cam_params['cam']['T']['data'])
    camera1_matrix = np.array(dual_cam_params['cam']['M1']['data']).reshape((3,3))
    camera2_matrix = np.array(dual_cam_params['cam']['M2']['data']).reshape((3,3))
    camera1_distortion = np.array(dual_cam_params['cam']['D1']['data'])
    camera2_distortion = np.array(dual_cam_params['cam']['D2']['data'])

    return R,T,camera1_matrix, camera1_distortion,camera2_matrix, camera2_distortion

def load_cam_params_txt(file_path):
    R = genfromtxt(file_path+'R.csv', delimiter=',')
    T = genfromtxt(file_path+'T.csv', delimiter=',')
    camera1_matrix = genfromtxt(file_path+'camera1_matrix.csv', delimiter=',')
    camera2_matrix = genfromtxt(file_path+'camera2_matrix.csv', delimiter=',')
    camera1_distortion = genfromtxt(file_path+'camera1_distortion.csv', delimiter=',')
    camera2_distortion = genfromtxt(file_path+'camera2_distortion.csv', delimiter=',')

    return R,T,camera1_matrix, camera1_distortion,camera2_matrix, camera2_distortion


def load_yaml_data(path):
    with open(path) as f_tmp:
        return yaml.load(f_tmp, Loader=yaml.FullLoader)
        # return yaml.load(f_tmp)


# Note dont use flags=cv2.CALIB_ZERO_DISPARITY when cx1 not equal to cx2
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
                            alpha=0
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

    f = Q[2, 3]
    baseline = 1./Q[3, 2]

    return map1_x, map1_y, map2_x, map2_y, f, baseline, Q, P1,P2

# file_path = '/media/deeplearner/PortableSSD/fibre_track/config.yaml'### calibration file path
# file_path = '/home/deeplearner/Documents/haozheng/fibre_track/data1015A/calibration/'
file_path = '/home/utsav/IProject/data/captured/newconfig.yaml'

R,T,m1, d1, m2, d2 = load_cam_params_dict(file_path)
# R,T,m1, d1, m2, d2 = load_cam_params_txt(file_path)


map1_x, map1_y, map2_x, map2_y, f, baseline, Q, P1, P2 = rectify(m1, d1, m2, d2, IMG_WIDTH, IMG_HEIGHT, R, T)
f_times_baseline = f * baseline

print('p1',P1)
print('p2',P2)
exp_idx = 'lnd1'
data_path = '/home/utsav/IProject/data/captured/'+ exp_idx
folder_name_camera_l = "rect_left"
folder_name_camera_r = "rect_right"

if not os.path.exists(os.path.join(data_path, folder_name_camera_l)):
    os.makedirs(os.path.join(data_path, folder_name_camera_l))
    os.makedirs(os.path.join(data_path, folder_name_camera_r))
# idx = 1
img_paths= os.listdir(data_path+"/left")
# for idx in range(3000):
for idx in range(len(img_paths)):
    if idx%100==0:
        print(idx)
    image1_path = os.path.join(data_path+"/left/{}.png".format(idx))
    im1 = cv2.imread(image1_path)
    image2_path = os.path.join(data_path+"/right/{}.png".format(idx))
    # image2_path = os.path.join(data_path+"/images/R/{}.png".format(idx))
    im2 = cv2.imread(image2_path)
    # im2 = cv2.imread(image1_path)
    
    # IMG_HEIGHT,IMG_WIDTH,_ =im1.shape

    # map1_x, map1_y, map2_x, map2_y, f, baseline, Q, P1, P2 = rectify(m1, d1, m2, d2, IMG_WIDTH, IMG_HEIGHT, R, T)
    # f_times_baseline = f * baseline

    im1_rect = cv2.remap(im1, map1_x, map1_y, cv2.INTER_LINEAR)
    im2_rect = cv2.remap(im2, map2_x, map2_y, cv2.INTER_LINEAR)

    # cv2.imwrite(os.path.join(data_path, folder_name_camera_l, '{}.png'.format(idx)), im1_rect)

    cv2.imwrite(os.path.join(data_path, folder_name_camera_l, '{}.png'.format(idx)), im1_rect)
    cv2.imwrite(os.path.join(data_path, folder_name_camera_r, '{}.png'.format(idx)), im2_rect)