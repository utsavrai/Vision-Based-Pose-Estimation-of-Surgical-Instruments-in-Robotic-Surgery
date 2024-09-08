import numpy as np
import cv2
from scipy.io import savemat
import scipy
import os
import yaml
from scipy.spatial.transform import Rotation as R
import glob
from get_pose import get_3d_obj_coordinate,load_image_and_undistort_it,get_2d_coordinates,get_pose
import matplotlib.pyplot as plt
from natsort import natsorted
from scipy.spatial.transform import Rotation as R
'''
Some Constant
'''
PATTERN_SHAPE = (3, 7)
DIST_BETWEEN_BLOBS_MM = 0.595*2 # [mm]

def angle_to_rot(angle_X,angle_Y,angle_Z):
    r = R.from_euler('xyz', [angle_X,angle_Y,angle_Z], degrees=True)
    return r.as_matrix()

def show_blob_detector_result(im, blob_detector):
    ## Find blob keypoints
    keypoints = blob_detector.detect(im) # mask_green_closed (try with that instead)
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    img_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Keypoints", img_with_keypoints)
    cv2.waitKey(0)
    return img_with_keypoints

def create_blob_detector():
    """ This function creates the blob detector with the same settings as Jian's paper """
    # TODO: hardcoded values
    params = cv2.SimpleBlobDetector_Params()
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
    detector = cv2.SimpleBlobDetector_create(params)
    return detector

def crate_idx(skip_idx_list):
    skip_idxs = []
    for skip_clip in skip_idx_list:
        if isinstance(skip_clip,int):
            if skip_clip not in skip_idxs:
                skip_idxs.append(skip_clip)
        else:
            skip_idx_seperate = skip_clip.split('-')
            sub_list = list(range(int(skip_idx_seperate[0]), int(skip_idx_seperate[1])+1))
            ### +1 last index should be included
            for sub_idx in sub_list:
                if sub_idx not in skip_idxs:
                    skip_idxs.append(sub_idx)
    return skip_idxs

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

def create_axis(length,num):
    a = np.linspace(0, length, num=num)
    x_pt  = np.zeros((num,3))
    x_pt[:,0] = a
    y_pt  = np.zeros((num,3))
    y_pt[:,1] = a
    z_pt  = np.zeros((num,3))
    z_pt[:,2] = a
    pts = np.concatenate((x_pt,y_pt,z_pt))
    cls = np.ones((3*num,3))
    cls[0:num,0]*=255           #R
    cls[num:2*num,1]*=255       #G
    cls[2*num:3*num,2]*=255     #B
    return pts,cls

def generate_skip_idx(skip_idx_list):
    skip_idxs = []
    for skip_clip in skip_idx_list:
        if isinstance(skip_clip,int):
            if skip_clip not in skip_idxs:
                skip_idxs.append(skip_clip)
        else:
            skip_idx_seperate = skip_clip.split('-')
            sub_list = list(range(int(skip_idx_seperate[0]), int(skip_idx_seperate[1])+1))
            ### +1 last index should be included
            for sub_idx in sub_list:
                if sub_idx not in skip_idxs:
                    skip_idxs.append(sub_idx)
    return skip_idxs

def show_mask(im, rvecs, tvecs, cam_matrix, dist_coeff, pts_3d):
    imgpts, jac = cv2.projectPoints(pts_3d, rvecs, tvecs, cam_matrix, dist_coeff)

    image_mask = np.zeros(im.shape[:2],dtype = np.uint8)
    radius = 1
    thickness = 1

    height,width,_ = im.shape

    for i, pt in enumerate(imgpts):
        pt_x = int(pt[0,0])
        pt_y = int(pt[0,1])
        # if pt_x<2000 and pt_x>-2000 and pt_y<2000 and pt_y>-2000:
        if pt_x<width and pt_x>-1 and pt_y<height and pt_y>-1:
            # use the BGR format to match the original image type
            cv2.circle(image_mask,(pt_x, pt_y), radius, 255, thickness)

    thresh = cv2.threshold(image_mask, 30, 255, cv2.THRESH_BINARY)[1]

    # contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    contours = contours[0]
    if contours is None:
        return None
    if len(contours)==0:
        return None

    cnt = max(contours, key=cv2.contourArea)

    image_mask = np.zeros(im.shape[:2],dtype = np.uint8)
    cv2.drawContours(image_mask, [cnt], -1, 255, -1)

    return image_mask

def show_fps(im, rvecs, tvecs, cam_matrix, dist_coeff, length, model_pts, fps_pts, is_show=False):

    pose_fps_2d, jac = cv2.projectPoints(fps_pts, rvecs, tvecs, cam_matrix, dist_coeff)
    pose_fps_2d_float = pose_fps_2d.copy()
    pose_fps_2d = np.rint(pose_fps_2d).astype(int)

    if is_show:
        im = im.copy()
        # image_mask = np.zeros(im.shape[:2],dtype = np.uint8)
        radius = 4
        thickness = 2

        for i, pt in enumerate(pose_fps_2d):
            pt_x = pt[0,0]
            pt_y = pt[0,1]
            cv2.circle(im, (pt_x, pt_y), radius, (255,100,100),thickness)
            # cv2.circle(image_mask,(pt_x, pt_y), radius, 255, thickness)

        cv2.imshow("image", im)
        cv2.waitKey(0)

    return pose_fps_2d, pose_fps_2d_float

def visualize_fps(im, rvecs, tvecs, cam_matrix, dist_coeff, length, model_pts, fps_pts, is_show=False): 
    pose_fps_2d, jac = cv2.projectPoints(fps_pts, rvecs, tvecs, cam_matrix, dist_coeff)
    pose_fps_2d_float = pose_fps_2d.copy().reshape(-1,2)
    pose_fps_2d = np.rint(pose_fps_2d.reshape(-1,2)).astype(int)
    print(pose_fps_2d.shape)

    plt.imshow(im) 
    plt.plot(pose_fps_2d[:, 0], pose_fps_2d[:, 1], '.') 
    plt.show()

def save_pose(im_path, mat):
    filename = '{}.txt'.format(im_path)
    print("save to ",filename)
    np.savetxt(filename, (mat), fmt="%s", delimiter=',')

def save_pose_np(im_path, mat):
    # print(mat)
    filename = '{}.npy'.format(im_path)
    np.save(filename, mat)

def show_axis(im, rvecs, tvecs, cam_matrix, dist_coeff, length, is_show=False):
    axis = np.float32([[0, 0, 0], [length,0,0], [0,length,0], [0,0,length]]).reshape(-1,3)
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, cam_matrix, dist_coeff)
    imgpts = np.rint(imgpts).astype(int)
    frame_centre = tuple(imgpts[0].ravel())

    thickness = 4
    im = cv2.line(im, frame_centre, tuple(imgpts[3].ravel()), (255,0,0), thickness, cv2.LINE_AA)#B 3 Z
    im = cv2.line(im, frame_centre, tuple(imgpts[2].ravel()), (0,255,0), thickness, cv2.LINE_AA)#G 2 Y
    im = cv2.line(im, frame_centre, tuple(imgpts[1].ravel()), (0,0,255), thickness, cv2.LINE_AA)#R 1 X

    if is_show:
        cv2.imshow("image", im)
        cv2.waitKey(0)

    return im

def show_blend_mask(im1, im2):
    if len(im2.shape)<3:
        im2 = cv2.cvtColor(im2,cv2.COLOR_GRAY2RGB)
    dst = cv2.addWeighted(im1, 1, im2, 0.2, 0)

    return dst

def show_axis2(im, rvecs, tvecs, cam_matrix, dist_coeff, length, is_show=False):
    long = 20
    axis = np.float32([[0, 0, 0], [long,0,0], [0,length,0], [0,0,length],[-long,0,0]]).reshape(-1,3)
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, cam_matrix, dist_coeff)
    imgpts = np.rint(imgpts).astype(int)
    frame_centre = tuple(imgpts[0].ravel())

    thickness = 4
    im = cv2.line(im, frame_centre, tuple(imgpts[3].ravel()), (255,0,0), thickness, cv2.LINE_AA)#B 3 Z
    im = cv2.line(im, frame_centre, tuple(imgpts[2].ravel()), (0,255,0), thickness, cv2.LINE_AA)#G 2 Y
    # im = cv2.line(im, frame_centre, tuple(imgpts[1].ravel()), (0,0,255), thickness, cv2.LINE_AA)#R 1 X
    im = cv2.line(im, tuple(imgpts[4].ravel()), tuple(imgpts[1].ravel()), (0,0,255), thickness, cv2.LINE_AA)#R 1 X

    if is_show:
        cv2.imshow("image", im)
        cv2.waitKey(0)

    return im

def show_3dmodel(im, rvecs, tvecs, cam_matrix, dist_coeff, pts_3d, is_show=False):
    imgpts, jac = cv2.projectPoints(pts_3d, rvecs, tvecs, cam_matrix, dist_coeff)
    frame_centre = tuple(imgpts[0].ravel())
    # thickness = 4

    radius = 1
    thickness = 1

    for i, pt in enumerate(imgpts):
        pt_x = int(pt[0,0])
        pt_y = int(pt[0,1])
        try:
            cv2.circle(im, (pt_x, pt_y), radius, (66,50,255),thickness)
        except:
            continue
    if is_show:
        cv2.imshow("image", im)
        cv2.waitKey(0)
    return im

def show_contour(im, rvecs, tvecs, cam_matrix, dist_coeff, pts_3d, is_show=False):
    imgpts, jac = cv2.projectPoints(pts_3d, rvecs, tvecs, cam_matrix, dist_coeff)
    image_mask = np.zeros(im.shape[:2],dtype = np.uint8)
    radius = 2
    thickness = 2

    height,width,_ = im.shape

    for i, pt in enumerate(imgpts):
        pt_x = int(pt[0,0])
        pt_y = int(pt[0,1])
        # if pt_x<2000 and pt_x>-2000 and pt_y<2000 and pt_y>-2000:
        if pt_x<width and pt_x>-1 and pt_y<height and pt_y>-1:
            # use the BGR format to match the original image type
            cv2.circle(image_mask,(pt_x, pt_y), radius, 255, thickness)

    thresh = cv2.threshold(image_mask, 30, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
    
    contours = contours[0]
    # print(contours)
    if contours is None:
        return im
    if len(contours)==0:
        return im
    
    cnt = max(contours, key=cv2.contourArea)
    image_mask = np.zeros(im.shape[:2],dtype = np.uint8)
    cv2.drawContours(im, [cnt], -1, (0,255,0), 1)
    # cv2.drawContours(im, [cnt], -1, (50,50,255), -1)
    if is_show:
        cv2.imshow("image", im)
        cv2.waitKey(0)

    return im

def create_filled_mask(im, rvecs, tvecs, cam_matrix, dist_coeff, pts_3d):
    imgpts, jac = cv2.projectPoints(pts_3d, rvecs, tvecs, cam_matrix, dist_coeff)
    image_mask = np.zeros(im.shape[:2], dtype=np.uint8)
    radius = 2
    thickness = 2

    height, width, _ = im.shape

    for i, pt in enumerate(imgpts):
        pt_x = int(pt[0, 0])
        pt_y = int(pt[0, 1])
        if pt_x < width and pt_x > -1 and pt_y < height and pt_y > -1:
            cv2.circle(image_mask, (pt_x, pt_y), radius, 255, thickness)

    thresh = cv2.threshold(image_mask, 30, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = contours[0]
    if contours is None or len(contours) == 0:
        return None

    cnt = max(contours, key=cv2.contourArea)
    image_mask = np.zeros(im.shape[:2], dtype=np.uint8)
    cv2.drawContours(image_mask, [cnt], -1, 255, -1)  # Fill the contour to create a mask

    return image_mask

def create_folder(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def generate_pose(root_path,data_path,config,camera1_matrix,camera2_matrix,img_format='.png'):

    ### Blob detector
    pts3d_mm = get_3d_obj_coordinate(DIST_BETWEEN_BLOBS_MM, PATTERN_SHAPE)
    blob_detector = create_blob_detector()

    camera_matrix_list = [camera1_matrix,camera2_matrix]
    folder_names = ['rect_left','rect_right']
    output_names = ['output_left','output_right']

    offset_rot = angle_to_rot(0,0,0)    ### Rotation Offset
    offset_trans = np.array([0,0,0])    ### Translatin Offset
    offset_homo = pack_homo(offset_rot,offset_trans) ### Offset Homography Matrix


    for i in range(2):
        camera_matrix = camera_matrix_list[i]
        folder_name = folder_names[i]
        output_name = output_names[i]
        create_folder(data_path+'mask_rgb/')
        create_folder(data_path+'patern_mask/')
        create_folder(data_path+'homo/')
        create_folder(data_path+output_name)


        img_paths=natsorted(glob.glob(data_path+'{}/*{}'.format(folder_name,img_format)))
        for img_idx in range(len(img_paths)):
            print('img_idx {}'.format(img_idx))
            valid = False

            img_path = img_paths[img_idx]
            im = cv2.imread(img_path, -1)

            dist_coefs = None 
            # blob_img = show_blob_detector_result(im, blob_detector)
            if im is None:
                continue

            pts2d = get_2d_coordinates(im, PATTERN_SHAPE, blob_detector)
            if pts2d is not None:
                valid, rvec_pred, tvec_pred = get_pose(pts3d_mm, pts2d, camera_matrix, dist_coefs, False, 5)

            if im is not None and valid:
                im_orig = im.copy()
                # cv2.imwrite(data_path+"check/{}.png".format(img_idx),im)
                # im = show_axis(blob_img, rvec_pred, tvec_pred, camera_matrix, dist_coefs, 6,True)
                rmat_pred, _ = cv2.Rodrigues(rvec_pred)
                marker_pose = pack_homo(rmat_pred,tvec_pred)

                ### Instrument Pose After Offset
                inst_pose = np.dot(marker_pose,offset_homo)
                r,t = unpack_homo(inst_pose)

                save_pose_np(data_path+"{}/{}".format(output_name,img_idx), marker_pose[:3])

def generate_data(root_path,data_path,config,camera_matrix,dist_coefs,img_format='.png',set_x0=False):
    from natsort import natsorted
    import re
    check_imgs_path = data_path+'check/'

    # skip_idx_list = config['skip_idx']
    # skip_idx = crate_idx(skip_idx_list)

    # pts_3d = np.load(root_path+'src/dvrk_model/PG/'+config['3d_model'])
    pts_3d = np.load(root_path+'dvrk_model/{}/'.format(subset_name)+'tool.npy')
    # sample_3d = np.load(root_path+'src/dvrk_model/{}/joint_sample1.npy'.format(subset_name))
    # fps_3d = np.load(root_path+'src/dvrk_model/PG/'+config['kp_model'])
    # fps_3d = np.load(data_path+config['kp_model'])
    rot2 = angle_to_rot(*config['pose_rot'])
    trans = np.array(config['pose_trans']) # RGB
    h1 = pack_homo(rot2,trans)


    img_paths=natsorted(glob.glob(data_path+'{}/*{}'.format('rect_left',img_format)))

    create_folder(data_path+'mask/')
    create_folder(data_path+'mask_depth/')
    create_folder(data_path+'mask_rgb/')
    create_folder(data_path+'final_pose/')
    create_folder(data_path+'filtered_left_rect_images/')

    img_idxs = []
    for img_path in img_paths:
        # img_idx = img_path[:-4]
        # img_idxs.append(int(img_idx))
        filename = os.path.basename(img_path)
    
        # Use a regular expression to find the numeric part of the filename
        match = re.search(r'(\d+)', filename)
        if match:
            img_idx = match.group(1)
            img_idxs.append(int(img_idx))

    for img_idx in img_idxs:
        # if img_idx in skip_idx:
        #     print("detect skip idx: {}".format(img_idx))
        #     continue
        img_path = data_path+'rect_left/{}{}'.format(img_idx,img_format)
       
        
        dist_coefs = None
        im = cv2.imread(img_path)
        
        

        homo_path = data_path+'output_left/{}.npy'.format(img_idx)
        if not os.path.isfile(homo_path):
            print("Skipping index {}: .npy file not found".format(img_idx))
            continue

        # depth_img_path = data_path+'depth_int32/{}{}'.format(img_idx,img_format)
        # dim = cv2.imread(depth_img_path)
        cv2.imwrite(data_path+"filtered_left_rect_images/{}{}".format(img_idx,img_format),im)
        h2 = np.load(homo_path)
        inst_pose = np.dot(h2,h1)
        r, t = unpack_homo(inst_pose)
        # im = show_axis(im, r, t, camera_matrix, dist_coefs, 6,False)        


        # im = show_3dmodel(im, r, t, camera_matrix,dist_coefs, sample_3d,False)
        im = show_3dmodel(im, r, t, camera_matrix,dist_coefs, pts_3d,False)
        im = show_contour(im, r, t, camera_matrix,dist_coefs, pts_3d,False)

        # dim = show_3dmodel(dim, r, t, camera_matrix,dist_coefs, pts_3d,False)
        # dim = show_contour(dim, r, t, camera_matrix,dist_coefs, pts_3d,False)
        
        mask_im = show_mask(im, r, t, camera_matrix,dist_coefs, pts_3d)
        # mask_im = create_filled_mask(im, r, t, camera_matrix, dist_coefs, pts_3d)
        cv2.imwrite(data_path + "mask/{}.png".format(img_idx), mask_im)
        # cv2.imwrite(data_path+"check/{}.jpg".format(img_idx),im)
        # cv2.imwrite(data_path+"mask/{}.png".format(img_idx),mask_im)
        np.save(data_path+'final_pose/{}.npy'.format(img_idx), inst_pose[:3,:])


        
        # cv2.imshow('Estimated Pose', im)
        # cv2.waitKey(0)
        # cv2.imshow('mask', mask_im)
        # cv2.waitKey(0)
        # cv2.imshow('blend', blend)
        # cv2.waitKey(0)
        if mask_im is not None:
            # cv2.imwrite(data_path+"mask/{}.png".format(img_idx),mask_im)
            cv2.imwrite(data_path+"mask_rgb/{}.jpg".format(img_idx),im)
            # cv2.imwrite(data_path+"mask_depth/{}.jpg".format(img_idx),dim)

        # cv2.imwrite(data_path+"check/{}.jpg".format(img_idx),check_im)
        # cv2.imwrite(data_path+"{}/{}_mask001.png".format(pattern_mask_folder_name,img_idx),mask)
        # cv2.imwrite(data_path+"{}/{}.jpg".format(mask_rgb_folder_name,img_idx),blend)

if __name__ == '__main__':
    global subset_name
    subset_name = 'LND'
    root_path = '/home/utsav/IProject/data/captured/'
    exp_idxs = ['lnd2']
    for exp_idx in exp_idxs:
        data_path = root_path+'{}/'.format(exp_idx)
        skip_path = data_path+'config.yaml'
        with open(skip_path) as f_tmp:
            config =  yaml.load(f_tmp, Loader=yaml.FullLoader)
        # camera_matrix = np.array(config['cam']['camera_matrix']['data']).reshape((3,3))
        # dist_coefs = np.array(config['cam']['dist_coeff']['data'])
        dist_coefs = None

        camera1_matrix = np.array(config['cam']['RECT_M1']['data']).reshape((3,3))
        camera2_matrix = np.array(config['cam']['RECT_M2']['data']).reshape((3,3))

        generate_pose(root_path,data_path,config['dataset'],camera1_matrix,camera2_matrix)
        generate_data(root_path,data_path,config['dataset'],camera1_matrix,dist_coefs)