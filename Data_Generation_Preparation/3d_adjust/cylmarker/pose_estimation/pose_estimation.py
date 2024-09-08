# from asyncio import FastChildWatcher
# from cylmarker import load_data, keypoints
# from cylmarker.pose_estimation import img_segmentation
# from scipy.io import savemat
import glob
import cv2 as cv
import numpy as np
import os.path
import scipy.io
from scipy.spatial.transform import Rotation as R
import math
from natsort import natsorted

def create_folder(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
def angle_to_rot(angle_X,angle_Y,angle_Z):
    r = R.from_euler('xyz', [angle_X,angle_Y,angle_Z], degrees=True)
    return r.as_matrix()
# def angle_to_rot(angle_X,angle_Y,angle_Z):
#     cost = np.cos(np.deg2rad(angle_Z))
#     sint = np.sin(np.deg2rad(angle_Z))
#     rot_Z = np.array([[cost, -sint, 0],
#                     [sint, cost, 0],
#                     [0, 0, 1]])
#     cost = np.cos(np.deg2rad(angle_X))
#     sint = np.sin(np.deg2rad(angle_X))
#     rot_X = np.array([[1, 0, 0],
#                     [0, cost, -sint],
#                     [0, sint, cost]])
#     rot = np.dot(rot_X,rot_Z)
#     cost = np.cos(np.deg2rad(angle_Y))
#     sint = np.sin(np.deg2rad(angle_Y))
#     rot_y = np.array([[cost, 0, sint],
#                         [0, 1, 0],
#                         [-sint, 0, cost]])
#     # rot = np.dot(rot,rot_y)
#     rot = np.dot(rot_Z, np.dot( rot_y, rot_X ))
#     return rot
# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta):

    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])

    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])

    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R

def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    # return np.array([x, y, z])
    return np.rad2deg([x, y, z])

def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix
def check_image(im, im_path):
    if im is None:
        print('Error opening the image {}'.format(im_path))
        # exit()
        return False
    else:
        return True





def make_rot(angle,axis='z'):
    cost = np.cos(np.deg2rad(angle))
    sint = np.sin(np.deg2rad(angle))
    if axis =='z':
        rot = np.array([[cost, -sint, 0],
                    [sint, cost, 0],
                    [0, 0, 1]])
    if axis =='x':
        rot = np.array([[1, 0, 0],
                    [0, cost, -sint],
                    [0, sint, cost]])
    if axis =='y':
        rot = np.array([[cost, 0, sint],
                    [0, 1, 0],
                    [-sint, 0, cost]])
    return rot

def show_contour(im, rvecs, tvecs, cam_matrix, dist_coeff, pts_3d, is_show=False):
    imgpts, jac = cv.projectPoints(pts_3d, rvecs, tvecs, cam_matrix, dist_coeff)
    image_mask = np.zeros(im.shape[:2],dtype = np.uint8)
    radius = 2
    thickness = 2


    for i, pt in enumerate(imgpts):
        pt_x = int(pt[0,0])
        pt_y = int(pt[0,1])
        # print((pt_x, pt_y))
        try:
            # cv2.circle(im, (pt_x, pt_y), radius, (100,100,255),thickness)
            cv.circle(image_mask,(pt_x, pt_y), radius, 255, thickness)
        except:
            continue

    thresh = cv.threshold(image_mask, 30, 255, cv.THRESH_BINARY)[1]
    # contours, _ = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
                                        # cv.CHAIN_APPROX_SIMPLE)
    contours = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    # print(contours[0])
    # input()
    cnt = max(contours[0], key=cv.contourArea)
    image_mask = np.zeros(im.shape[:2],dtype = np.uint8)
    cv.drawContours(im, [cnt], -1, (0,255,0), 1)
    if is_show:
        cv.imshow("image", im)
        cv.waitKey(0)

    return im

def show_axis(im, rvecs, tvecs, cam_matrix, dist_coeff, length, is_show=False):
    axis = np.float32([[0, 0, 0], [length,0,0], [0,length,0], [0,0,length]]).reshape(-1,3)
    imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, cam_matrix, dist_coeff)
    imgpts = np.rint(imgpts).astype(int)
    frame_centre = tuple(imgpts[0].ravel())

    thickness = 4
    im = cv.line(im, frame_centre, tuple(imgpts[3].ravel()), (255,0,0), thickness, cv.LINE_AA)#B 3 Z
    im = cv.line(im, frame_centre, tuple(imgpts[2].ravel()), (0,255,0), thickness, cv.LINE_AA)#G 2 Y
    im = cv.line(im, frame_centre, tuple(imgpts[1].ravel()), (0,0,255), thickness, cv.LINE_AA)#R 1 X

    if is_show:
        cv.imshow("image", im)
        cv.waitKey(0)

    return im

def show_axis_align(im, rvecs, tvecs, cam_matrix, dist_coeff, length, is_show=False):
    axis = np.float32([[0, 0, 0], [length,0,0], [0,length,0], [0,0,length],[0,-length,0]]).reshape(-1,3)
    imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, cam_matrix, dist_coeff)
    imgpts = np.rint(imgpts).astype(int)
    frame_centre = tuple(imgpts[0].ravel())

    thickness = 4
    im = cv.line(im, frame_centre, tuple(imgpts[3].ravel()), (255,0,0), thickness, cv.LINE_AA)#B 3 Z
    im = cv.line(im, frame_centre, tuple(imgpts[2].ravel()), (0,255,0), thickness, cv.LINE_AA)#G 2 Y
    im = cv.line(im, frame_centre, tuple(imgpts[1].ravel()), (0,0,255), thickness, cv.LINE_AA)#R 1 X

    im = cv.line(im, frame_centre, tuple(imgpts[4].ravel()), (0,255,0), thickness, cv.LINE_AA)#B 3 Z

    if is_show:
        cv.imshow("image", im)
        cv.waitKey(0)

    return im

def show_square(im, rvecs, tvecs, cam_matrix, dist_coeff, length):
    off_x, off_y = -1.5,-2
    axis = np.float32([[off_x, off_y, 0], [off_x+length,off_y,0], [off_x,off_y+length,0],[off_x+length,off_y+length,0]]).reshape(-1,3)
    # axis = np.float32([[0, 0, 0], [length,0,0], [0,length,0],]).reshape(-1,3)
    imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, cam_matrix, dist_coeff)
    imgpts = np.rint(imgpts).astype(int)
    frame_centre = tuple(imgpts[0].ravel())

    thickness = 2
    im = cv.line(im, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (255,0,0), thickness, cv.LINE_AA)
    im = cv.line(im, tuple(imgpts[0].ravel()), tuple(imgpts[2].ravel()), (255,0,0), thickness, cv.LINE_AA)
    im = cv.line(im, tuple(imgpts[3].ravel()), tuple(imgpts[1].ravel()), (255,0,0), thickness, cv.LINE_AA)
    im = cv.line(im, tuple(imgpts[3].ravel()), tuple(imgpts[2].ravel()), (255,0,0), thickness, cv.LINE_AA)

    return im

def show_3dmodel_mask(im, rmats, tvecs, cam_matrix, dist_coeff, pts_3d, is_show=False):
    im = im.copy()
    rot2 = make_rot(0,'y')
    # rmats, _ = cv.Rodrigues(rvecs)
    rmats2 = np.dot(rmats,rot2)
    tvecs1 = tvecs.copy()
    imgpts, jac = cv.projectPoints(pts_3d, rmats2, tvecs1, cam_matrix, dist_coeff)
    frame_centre = tuple(imgpts[0].ravel())
    # thickness = 4

    image_mask = np.zeros(im.shape[:2],dtype = np.uint8)
    radius = 1
    thickness = 1

    for i, pt in enumerate(imgpts):
        pt_x = int(pt[0,0])
        pt_y = int(pt[0,1])
        # use the BGR format to match the original image type
        # col = (colors[i, 2], colors[i, 1], colors[i, 0])
        # cv.circle(im, (pt_x, pt_y), 5, (100,100,255))
        # cv.circle(image_mask,(pt_x, pt_y), 5, 255, -1)
        try:
            cv.circle(im, (pt_x, pt_y), radius, (100,100,255),thickness)
            cv.circle(image_mask,(pt_x, pt_y), radius, 255, thickness)
        except:
            print("out of range")

    thresh = cv.threshold(image_mask, 30, 255, cv.THRESH_BINARY)[1]
    contours, _ = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
                                        cv.CHAIN_APPROX_SIMPLE)
    # print("contours:",contours)
    cnt = max(contours, key=cv.contourArea)
    image_mask = np.zeros(im.shape[:2],dtype = np.uint8)
    cv.drawContours(image_mask, [cnt], -1, 255, -1)

    if is_show:
        cv.imshow("image", im)
        cv.waitKey(0)

    return im
def show_3dmodel(im, rvecs, tvecs, cam_matrix, dist_coeff, pts_3d, is_show=False):
    imgpts, jac = cv.projectPoints(pts_3d, rvecs, tvecs, cam_matrix, dist_coeff)
    # frame_centre = tuple(imgpts[0].ravel())
    # thickness = 4

    image_mask = np.zeros(im.shape[:2],dtype = np.uint8)
    radius = 1
    thickness = 1

    for i, pt in enumerate(imgpts):
        pt_x = int(pt[0,0])
        pt_y = int(pt[0,1])
        # print((pt_x, pt_y))
        try:
            # cv2.circle(im, (pt_x, pt_y), radius, (100,100,255),thickness)
            cv.circle(im, (pt_x, pt_y), radius, (66,50,255),thickness)
            # cv2.circle(image_mask,(pt_x, pt_y), radius, 255, thickness)
        except:
            continue
    if is_show:
        cv.imshow("image", im)
        cv.waitKey(0)
    return im

def show_fps(im, rvecs, tvecs, cam_matrix, dist_coeff, length, model_pts, fps_pts):
    rot2 = make_rot(0,'y')
    rmats, _ = cv.Rodrigues(rvecs)
    rmats2 = np.dot(rmats,rot2)
    tvecs1 = tvecs.copy()

    pose_fps_2d, jac = cv.projectPoints(fps_pts, rmats2, tvecs1, cam_matrix, dist_coeff)
    pose_fps_2d_float = pose_fps_2d.copy()
    pose_fps_2d = np.rint(pose_fps_2d).astype(int)

    image_mask = np.zeros(im.shape[:2],dtype = np.uint8)
    radius = 2
    thickness = 2

    for i, pt in enumerate(pose_fps_2d):
        pt_x = pt[0,0]
        pt_y = pt[0,1]
        cv.circle(im, (pt_x, pt_y), radius, (255,100,100),4)
        cv.circle(image_mask,(pt_x, pt_y), radius, 255, 4)

    # cv.imshow("image", im)
    # cv.waitKey(0)

    return pose_fps_2d, pose_fps_2d_float

def show_mask(im, rvecs, tvecs, cam_matrix, dist_coeff, length, pts_3d):
    imgpts, jac = cv.projectPoints(pts_3d, rvecs, tvecs, cam_matrix, dist_coeff)
    frame_centre = tuple(imgpts[0].ravel())

    image_mask = np.zeros(im.shape[:2],dtype = np.uint8)
    radius = 3
    thickness = 2

    for i, pt in enumerate(imgpts):
        pt_x = int(pt[0,0])
        pt_y = int(pt[0,1])
        # use the BGR format to match the original image type
        cv.circle(image_mask,(pt_x, pt_y), radius, 255, thickness)

    thresh = cv.threshold(image_mask, 30, 255, cv.THRESH_BINARY)[1]
    contours, _ = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
                                        cv.CHAIN_APPROX_SIMPLE)
    # print("contours:",contours)
    cnt = max(contours, key=cv.contourArea)
    image_mask = np.zeros(im.shape[:2],dtype = np.uint8)
    cv.drawContours(image_mask, [cnt], -1, 255, -1)

    # cv.imshow("mask", image_mask)
    # cv.waitKey(0)

    return image_mask


def get_transf_inv(transf):
    # ref: https://math.stackexchange.com/questions/152462/inverse-of-transformation-matrix
    r = transf[0:3, 0:3]
    t = transf[0:3, 3]
    r_inv = np.transpose(r) # Orthogonal matrix so the inverse is its transpose
    t_new = np.matmul(-r_inv, t).reshape(3, 1)
    transf_inv = np.concatenate((r_inv, t_new), axis = 1)
    return transf_inv


def save_pts_info(im_path, pnts_3d_object, pnts_2d_image):
    filename = '{}.pts3d.txt'.format(im_path)
    np.savetxt(filename, (np.squeeze(pnts_3d_object)), fmt="%s", delimiter=',')
    filename = '{}.pts2d.txt'.format(im_path)
    np.savetxt(filename, (np.squeeze(pnts_2d_image)), fmt="%s", delimiter=',')


def save_pose(im_path, mat):
    filename = '{}.txt'.format(im_path)
    print("save to ",filename)
    np.savetxt(filename, (mat), fmt="%s", delimiter=',')

def save_pose_np(im_path, mat):
    # print(mat)
    filename = '{}.npy'.format(im_path)
    np.save(filename, mat)

# def show_reproj_error_image(im, pts2d_filtered, pts2d_projected):
#     red = (0, 0, 255)
#     green = (0, 255, 0)
#     for pt2d_d, pt2d_p in zip(pts2d_filtered[:,0], pts2d_projected[:,0]):
#         pt2d_d = (int(round(pt2d_d[0])), int(round(pt2d_d[1])))
#         pt2d_p = (int(round(pt2d_p[0])), int(round(pt2d_p[1])))
#         im = cv.line(im, pt2d_d, pt2d_p, color=red, thickness=1, lineType=cv.LINE_AA)
#         im = cv.circle(im, pt2d_d, radius=1, color=red, thickness=-1)
#         im = cv.circle(im, pt2d_p, radius=1, color=green, thickness=-1)
#     cv.imshow("image", im)
#     cv.waitKey(0)


def draw_detected_and_projected_features(rvecs, tvecs, cam_matrix, dist_coeff, pttrn, im):
    for sqnc in pttrn.list_sqnc:
        if sqnc.sqnc_id != -1:
            """
                We will draw the detected and projected contours of each feature in a sequence.
            """
            for kpt in sqnc.list_kpts:
                # First, we draw the detected contour (in green)
                cntr = kpt.cntr
                #im = cv.drawContours(im, [cntr], -1, [0, 255, 0], -1)
                im = cv.drawContours(im, [cntr], -1, [0, 255, 0], 1)
                # Then, we calculate + draw the projected contour (in red)
                corners_3d = np.float32(kpt.xyz_corners).reshape(-1,3)
                imgpts, jac = cv.projectPoints(corners_3d, rvecs, tvecs, cam_matrix, dist_coeff)
                imgpts = np.asarray(imgpts, dtype=np.int32)
                #im = cv.fillPoly(im, [imgpts], [0, 0, 255])
                im = cv.polylines(im, [imgpts], True, [0, 0, 255], thickness=1, lineType=cv.LINE_AA)
    return im

def read_trans_mat(mat_path):
    mat = scipy.io.loadmat(mat_path)
    Homo = mat['data'][0]
    Trans = Homo[:3].reshape(3,1)
    Rot = Homo[3:]
    B_r = quaternion_rotation_matrix(Rot)
    B = np.concatenate((B_r, Trans),axis=1)
    end = np.array([[0,0,0,1]])
    B = np.concatenate((B, end),axis=0)
    return B

title_window = 'Adjust 3D model!'
img_paths = []
rot_x = 0
rot_y = 0
rot_z = 0
trans_x = 0
trans_y = 0
trans_z = 0
im_ind = 0
mouse_x = 0
mouse_y = 0
config_file_data_copy = None
data_pttrn =None
data_marker =None

def pack_homo(R,T):
    T = T.reshape(3,-1)
    homo = np.concatenate((R, T), axis = 1)
    ones = np.array([0,0,0,1]).reshape(1,4)
    homo = np.concatenate((homo, ones), axis = 0)
    return homo

def unpack_homo(homo):
    R = homo[:,:3][:3]
    t = homo[:,-1][:3]
    return R,t

def trackbar_callback_im(im_ind_new):
    global im_ind
    global img_folder, pose_folder
    im_ind = im_ind_new
    pose_path = pose_paths[im_ind]
    im_path = pose_path.replace(pose_folder,img_folder)
    im_path = im_path.replace('npy','png')
    # print(im_path.split('/'))
    im_id = im_path.split('/')[-1][:-4]
    # im_path = img_paths[im_ind]
    im = cv.imread(im_path, cv.IMREAD_COLOR)
    check_image(im, im_path) # check if image was sucessfully read
    dist_coeff = None

    rot = angle_to_rot(rot_x,rot_y,rot_z)
    # rot = eulerAnglesToRotationMatrix([rot_x,rot_y,rot_z])
    trans = np.array([trans_x,trans_y,trans_z]) # RGB
    text_content = "id: {}|trans: ({:.2f},{:.2f},{:.2f})| rot: ({:.2f},{:.2f},{:.2f})".format(im_id, trans_x,trans_y,trans_z,rot_x,rot_y,rot_z)
    h1 = pack_homo(rot,trans)
    
    # pose_path = im_path.replace('undistort','pattern_pose')
    # pose_path = pose_path.replace('png','npy')
    # print(pose_path)

    pose = np.load(pose_path)

    rmat_pred, tvec_pred = unpack_homo(pose)

    im = show_axis(im, rmat_pred, tvec_pred, cam_matrix, dist_coeff, 6,False)
    im = show_square(im, rmat_pred, tvec_pred, cam_matrix, dist_coeff, 10)

    h2 = pack_homo(rmat_pred,tvec_pred)
    h3 = np.dot(h2,h1)
    rvec_pred,tvec_pred = unpack_homo(h3)

    text_pos = (100,100)
    font = cv.FONT_HERSHEY_SIMPLEX
    fontScale=0.5
    color = (255,255,255)
    thickness = 2

    im = cv.putText(im, text_content, text_pos, font, 
                   fontScale, color, thickness, cv.LINE_AA)


    rot_marker = angle_to_rot(0,0,-90)
    trans_marker = np.array([0,0,0]) # RGB
    h_marker = pack_homo(rot_marker,trans_marker)
    h_marker = np.dot(h3,h_marker)
    rot_marker,trans_marker = unpack_homo(h_marker)


    im = show_axis(im, rvec_pred, tvec_pred, cam_matrix, dist_coeff, 6,False)
    # im = show_axis_align(im, rvec_pred, tvec_pred, cam_matrix, dist_coeff, 15,False)
    im = show_3dmodel(im, rot_marker, trans_marker, cam_matrix, dist_coeff, pts_3d, is_show=False)
    # im = show_3dmodel(im, rvec_pred, tvec_pred, cam_matrix, dist_coeff, pts_3d, is_show=False)
    # im = show_3dmodel(im, rvec_pred, tvec_pred, cam_matrix, dist_coeff, contour_3d, is_show=False)
    im = show_contour(im, rvec_pred, tvec_pred, cam_matrix, dist_coeff, contour_3d, is_show=False)
    cv.imshow(title_window, im)


def trackbar_callback_x_trans(trans_X_new,width=3):
    global trans_x
    trans_X_new = (2*width)*(trans_X_new/100)+itrans_x-width
    trans_x = trans_X_new
    trackbar_callback_im(im_ind)

def trackbar_callback_y_trans(trans_Y_new,width=3):
    global trans_y
    trans_Y_new = (2*width)*(trans_Y_new/100)+itrans_y-width
    trans_y = trans_Y_new
    trackbar_callback_im(im_ind)

def trackbar_callback_z_trans(trans_Z_new,width=3):
    global trans_z
    trans_Z_new = (2*width)*(trans_Z_new/100)+itrans_z-width
    trans_z = trans_Z_new
    trackbar_callback_im(im_ind)

def trackbar_callback_x_rot(rot_X_new, width=30):
    global rot_x
    rot_X_new = (2*width)*(rot_X_new/100)+irot_x-width
    rot_x = rot_X_new
    trackbar_callback_im(im_ind)

def trackbar_callback_y_rot(rot_Y_new, width=30):
    global rot_y
    rot_Y_new = (2*width)*(rot_Y_new/100)+irot_y-width
    rot_y = rot_Y_new
    trackbar_callback_im(im_ind)

def trackbar_callback_z_rot(rot_Z_new, width=30):
    global rot_z
    rot_Z_new = (2*width)*(rot_Z_new/100)+irot_z-width
    rot_z = rot_Z_new
    trackbar_callback_im(im_ind)

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    mouse_x = x
    mouse_y = y
    trackbar_callback_im(im_ind)

def improve_segmentation(root,data_path,config,camera_matrix,dist_coefs,img_format='.png'):
    global trans_x, trans_y, trans_z
    global rot_x,rot_y,rot_z
    # global img_paths
    global pose_paths
    global dist_coeff_np,cam_matrix
    global pts_3d, contour_3d
    global root_path
    global itrans_x, itrans_y, itrans_z
    global irot_x, irot_y, irot_z
    global img_folder, pose_folder
    root_path = root

    side = 'left'

    img_folder = 'rect_{}'.format(side)
    pose_folder = 'output_{}'.format(side)
    # rot_x, rot_y, rot_z = config['pose_rot']
    # trans_x, trans_y, trans_z = config['pose_trans']

    # irot_x, irot_y, irot_z = config['pose_rot']
    # itrans_x, itrans_y, itrans_z = config['pose_trans']

    # pose_trans =  [-17.06,-1.0,1.17]
    # pose_rot =  [0.6,0.0,-91.2]


    pose_trans =  [-15.00, -4.00, 0.33]
    pose_rot =  [3,0,-91.2]

    # pose_trans =  [-17.48, -0.76, 0.09]
    # pose_rot =  [3,0,-90]

    rot_x, rot_y, rot_z = pose_rot
    trans_x, trans_y, trans_z = pose_trans

    irot_x, irot_y, irot_z = pose_rot
    itrans_x, itrans_y, itrans_z = pose_trans

    print("itrany", itrans_y)


    cam_matrix = camera_matrix
    dist_coeff_np = dist_coefs
    img_dir_path = data_path
    # img_paths = glob.glob(img_dir_path + 'undistort/*{}'.format(img_format))
    print(img_dir_path)
    pose_paths = natsorted(glob.glob(img_dir_path + '/output_left/*.npy'))
    print("num of pose",len(pose_paths))

    model_folder = 'path_to_dvrk_model/LND' #adjust accordingly
    

    pts_3d = np.load(root_path+'{}/tool.stl'.format(model_folder))
    contour_3d = np.load(root_path+'{}/tool.npy'.format(model_folder))

    # contour_3d = np.load(root_path+'src/dvrk_model/{}/'.format(model_folder)+config['3d_model'])
    # contour_3d = np.load(root_path+'src/dvrk_model/{}/joint.npy'.format(model_folder))
    # pts_3d = np.load(root_path+'src/dvrk_model/{}/joint_sample1.npy'.format(model_folder))

    # contour_3d = np.load(root_path+'/dvrk_model/{}/LND_cut_notip_dense.npy'.format(model_folder))
    # pts_3d = np.load(root_path+'/dvrk_model/{}/joint_sample1.npy'.format(model_folder))

    print('Press any [key] when finished.')

    cv.namedWindow(title_window, cv.WINDOW_NORMAL)
    cv.createTrackbar("Image", title_window , 0, len(pose_paths)-1, trackbar_callback_im)
    cv.createTrackbar("Trans X", title_window , 50, 100, trackbar_callback_x_trans)
    cv.createTrackbar("Trans Y", title_window , 50, 100, trackbar_callback_y_trans)
    cv.createTrackbar("Trans Z", title_window , 50, 100, trackbar_callback_z_trans)
    cv.createTrackbar("Rot X", title_window , 50, 100, trackbar_callback_x_rot)
    cv.createTrackbar("Rot Y", title_window , 50, 100, trackbar_callback_y_rot)
    cv.createTrackbar("Rot Z", title_window , 50, 100, trackbar_callback_z_rot)
    if len(pose_paths) > 0:
        trackbar_callback_im(im_ind)
        # cv.setMouseCallback(title_window, mouse_callback)
        cv.waitKey()
    else:
        print('ERROR: No images found')
    
    print('fianl parameter is:')
    print('rot: {}, {}, {}'.format(rot_x, rot_y, rot_z))
    print('trans: {}, {}, {}'.format(trans_x, trans_y, trans_z))

'''
def trackbar_callback_x_trans(trans_X_new,v_min=-10,v_max=-6):
    global trans_x
    trans_X_new = (v_max-v_min)*(trans_X_new/100)+v_min
    trans_x = trans_X_new
    trackbar_callback_im(im_ind)

def trackbar_callback_y_trans(trans_Y_new,v_min=-30,v_max=-25):
    global trans_y
    trans_Y_new = (v_max-v_min)*(trans_Y_new/100)+v_min
    trans_y = trans_Y_new
    trackbar_callback_im(im_ind)

def trackbar_callback_z_trans(trans_Z_new,v_min=0,v_max=3):
    global trans_z
    trans_Z_new = (v_max-v_min)*(trans_Z_new/100)+v_min
    trans_z = trans_Z_new
    trackbar_callback_im(im_ind)

def trackbar_callback_x_rot(rot_X_new, v_min=-30,v_max=30):
    global rot_x
    rot_X_new = (v_max-v_min)*(rot_X_new/100)+v_min
    rot_x = rot_X_new
    trackbar_callback_im(im_ind)

def trackbar_callback_y_rot(rot_Y_new, v_min=-30,v_max=30):
    global rot_y
    rot_Y_new = (v_max-v_min)*(rot_Y_new/100)+v_min
    rot_y = rot_Y_new
    trackbar_callback_im(im_ind)

def trackbar_callback_z_rot(rot_Z_new, v_min=-270,v_max=-90):
    global rot_z
    rot_Z_new = (v_max-v_min)*(rot_Z_new/100)+v_min
    rot_z = rot_Z_new
    trackbar_callback_im(im_ind)
'''
