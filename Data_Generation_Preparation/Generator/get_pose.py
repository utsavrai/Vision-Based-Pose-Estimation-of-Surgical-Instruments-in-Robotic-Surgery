import os
import yaml
from scipy.io import savemat
import cv2 as cv
import numpy as np

def unpack_homo(homo):
    R = homo[:,:3][:3]
    t = homo[:,-1][:3]
    return R,t

def save_pose_mat(im_path, mat):
    filename = '{}.mat'.format(im_path)
    savemat(filename, {'data': mat})

def save_pose_np(im_path, mat):
    filename = '{}.npy'.format(im_path)
    np.save(filename, mat)

def get_3d_obj_coordinate(d, PATTERN_SHAPE):
    """ This function gets the 3D coordinates of the points

        hexagonal grid pattern:

                       d
                      |-|
            x   x   x   x -
              x   x   x   - d
            x   x   x   x
              x   x   x
            x   x   x   x
              x   x   x

        d = DIST_BETWEEN_BLOBS
    """
    pts3d = []
    z = 0 # all points in the same plane
    for i in range(PATTERN_SHAPE[1]):
        y_shift = d
        if i % 2 == 0:
            y_shift = 0
        for j in range(PATTERN_SHAPE[0]):
            """ i =
                    0   2   4   6
                      1   3   5
                    x   x   x   x
                      x   x   x
                    x   x   x   x
                      x   x   x
            """
            x = i * d
            y = j * d * 2.0
            pts3d.append([[x, y + y_shift, z]])
    pts3d = np.asarray(pts3d, dtype=float)
    return pts3d


def create_blob_detector():
    """ This function creates the blob detector with the same settings as Jian's paper """
    # TODO: hardcoded values
    params = cv.SimpleBlobDetector_Params()
    params.minThreshold = 20
    params.maxThreshold = 220
    params.minDistBetweenBlobs = 5
    params.filterByArea = True
    params.minArea = 10
    params.maxArea = 1000
    params.filterByInertia = True
    params.minInertiaRatio = 0.1
    params.filterByConvexity = True
    params.minConvexity = 0.95
    params.filterByColor = True
    params.blobColor = 0
    params.minRepeatability = 2
    detector = cv.SimpleBlobDetector_create(params)
    return detector


def is_path_file(string):
    """ This function checks if the file path exists """
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), string)


def load_yaml_data(path):
    """ This function loads data from a yaml file """
    if is_path_file(path):
        with open(path) as f_tmp:
            return yaml.load(f_tmp, Loader=yaml.FullLoader)


def load_camera_parameters(cam_calib_file):
    """ This function loads the intrinsic and distortion camera parameters """
    if is_path_file(cam_calib_file):
        cam_calib_data = load_yaml_data(cam_calib_file)
        cam_matrix = cam_calib_data['camera_matrix']['data']
        cam_matrix = np.reshape(cam_matrix, (3, 3))
        dist_coeff = cam_calib_data['dist_coeff']['data']
        dist_coeff = np.array(dist_coeff)
    return cam_matrix, dist_coeff


def load_image_and_undistort_it(im_path, cam_matrix, dist_coeff):
    """ This function load an image and undistorts it """
    if is_path_file(im_path):
        im = cv.imread(im_path, -1)
        im = cv.undistort(im, cam_matrix, dist_coeff)
    return im


def fix_pal_effect(im):
    height, width, _ = im.shape
    im_odd = im[::2, :]  # odd rows
    im = cv.resize(im_odd, (width, height), interpolation=cv.INTER_CUBIC)
    return im

def show_blob_detector_result(im, blob_detector):
    ## Find blob keypoints
    keypoints = blob_detector.detect(im) # mask_green_closed (try with that instead)
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    img_with_keypoints = cv.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imshow("Keypoints", img_with_keypoints)
    cv.waitKey(0)


def get_2d_coordinates(im, PATTERN_SHAPE, blob_detector):
    ret, corners = cv.findCirclesGrid(im, PATTERN_SHAPE, flags=(cv.CALIB_CB_ASYMMETRIC_GRID + cv.CALIB_CB_CLUSTERING), blobDetector=blob_detector)
    if ret:
        return corners
    return None


# def show_axis(im, rvecs, tvecs, cam_matrix, dist_coeff, length):
#     #print(cam_matrix)
#     #print(np.transpose(tvecs))
#     axis = np.float32([[0, 0, 0], [length,0,0], [0,length,0], [0,0,length]]).reshape(-1,3)
#     #print(axis)
#     imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, cam_matrix, dist_coeff)
#     #print(imgpts)
#     frame_centre = tuple(imgpts[0].ravel())

#     thickness = 3
#     im = cv.line(im, frame_centre, tuple(imgpts[3].ravel()), (255,0,0), thickness, cv.LINE_AA)
#     im = cv.line(im, frame_centre, tuple(imgpts[2].ravel()), (0,255,0), thickness, cv.LINE_AA)
#     im = cv.line(im, frame_centre, tuple(imgpts[1].ravel()), (0,0,255), thickness, cv.LINE_AA)

#     cv.imshow("image", im)
#     cv.waitKey(0)

#     return im


def show_axis(im, transf, rvecs, tvecs, cam_matrix, dist_coeff, length_m):
    axis = np.float32([[0, 0, 0], [length_m,0,0], [0,length_m,0], [0,0,length_m]]).reshape(-1,3)
    # Understand which axis to draw first
    dist_z = []
    for target_T_pt in axis[1:]:
        target_T_pt = np.vstack((target_T_pt.reshape(3,1), [1]))
        cam_T_pt = np.matmul(transf, target_T_pt)
        dist_z.append(-cam_T_pt[2,0]) # - so that we sort from further to closer
    args = np.argsort(dist_z)
    imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, cam_matrix, dist_coeff)
    frame_centre = tuple(np.rint(imgpts[0]).astype(int).ravel())
    thickness = 4
    colors = [(0,0,255), (0,255,0), (255,0,0)] # Red, Green, Blue
    for i in args:
        cv.line(im, frame_centre, tuple(np.rint(imgpts[i+1]).astype(int).ravel()), colors[i], thickness, cv.LINE_AA)
    # cv.imshow("image", im)
    # cv.waitKey(500)
    return im


def get_pose(pts3d_mm, pts2d, cam_mat, dist, BOOL_SHOW_AXIS, AXIS_LENGTH):
    valid, rvec_pred, tvec_pred = cv.solvePnP(pts3d_mm, pts2d, cam_mat, dist)
    if valid and BOOL_SHOW_AXIS:
        # print(tvec_pred)
        # print("show image")
        # show_axis(im, rvec_pred, tvec_pred, cam_mat, dist, AXIS_LENGTH)
        transf = np.concatenate((rvec_pred, tvec_pred), axis = 1)
        show_axis(im, transf, rvec_pred, tvec_pred, cam_mat, dist, AXIS_LENGTH)
    return valid, rvec_pred, tvec_pred#, inliers

if __name__ == "__main__":
    from os import path
    from scipy.spatial.transform import Rotation as R

    DIST_BETWEEN_BLOBS_MM = 0.595*2 # [mm]
    CAM_CALIB = 'camera_calibration.yaml'
    PATTERN_SHAPE = (3, 7)
    BOOL_SHOW_AXIS = True
    AXIS_LENGTH = 4

    pts3d_mm = get_3d_obj_coordinate(DIST_BETWEEN_BLOBS_MM, PATTERN_SHAPE)
    blob_detector = create_blob_detector()
    cam_mat, dist = load_camera_parameters(CAM_CALIB)
    print(cam_mat)

    exp_idx = 'exp_1031markcali2'
    img_dir_path = 'F:/micc_challenge/0328_test/pg/left_resize/'
    # img_dir_path = '/media/deeplearner/PortableSSD/dvrk_pose/PG/{}/left_resize/'.format(exp_idx)
    # print(img_dir_path)
    axis_path = img_dir_path+'axis/'
    axis_both_path = img_dir_path+'axis_both/'
    data_path = img_dir_path+'masked/'
    # quat_list = []
    if not os.path.exists(img_dir_path+'homo_keydots/'):
        os.makedirs(img_dir_path+'homo_keydots/')
    if not os.path.exists(axis_both_path):
        os.makedirs(axis_both_path)
    for idx in range(350,360):
        valid = False
        IMG_PATH = data_path+'{}.png'.format(idx)
        # print(IMG_PATH)
        if path.exists(IMG_PATH):
            im = cv.imread(IMG_PATH, -1)
        else:
            continue

        # im = load_image_and_undistort_it(IMG_PATH, cam_mat, dist)
        dist = None # we don't need to undistort again

        # show_blob_detector_result(im, blob_detector)
        pts2d = get_2d_coordinates(im, PATTERN_SHAPE, blob_detector)
        if pts2d is not None:
            valid, rvec_pred, tvec_pred = get_pose(pts3d_mm, pts2d, cam_mat, dist, BOOL_SHOW_AXIS, AXIS_LENGTH)
        if valid:
            rmat_pred, _ = cv.Rodrigues(rvec_pred)
            # r = R.from_matrix(rmat_pred.reshape(3,3))
            # r_quat = r.as_quat()
            # if len(quat_list)>0:
            #     print('current quat',r_quat)
            #     print("last quat",quat_list[-1])
            #     diff = r_quat*quat_list[-1]
            #     print('multiplication',diff)
            #     print("--------------------------------------")
            # quat_list.append(r_quat)
            transf = np.concatenate((rmat_pred, tvec_pred), axis = 1)
            # print(transf)
            # save_pose_mat(img_dir_path+"homo_keydots/{}".format(idx), transf)
            save_pose_np(img_dir_path+"homo_keydots/{}".format(idx), transf)
            AXIS_PATH = axis_path+'{}.png'.format(idx)
            if path.exists(AXIS_PATH):
                print(AXIS_PATH)
                im = cv.imread(AXIS_PATH, -1)
                im = show_axis(im, transf, rvec_pred, tvec_pred, cam_mat, dist, AXIS_LENGTH)
                cv.imwrite(axis_both_path+'{}.png'.format(idx),im)

