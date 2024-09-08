from cylmarker import load_data
from cylmarker.pose_estimation import pose_estimation
import argparse
import yaml
import numpy as np
import os

def get_args():
    parser = argparse.ArgumentParser(description='Evaluateion script for SurgRIPE.')
    parser.add_argument('--path', help= 'Get path to data root path.')
    parser.add_argument('--side', type=str, default='l', help= 'Instrument Type for test.')
    # parser.add_argument('--spath', help= 'Get save path of results.')
    return parser.parse_args()

def main():
    args = get_args()
    side = args.side

    root_path = args.path

    # root_path = '/media/deeplearner/PortableSSD/msc_data/'
    # root_path = 'F:/dvrk_pose/'
    # subset_name = 'lnd1'
    exp_idxs = ['lnd1'] # dataset folder name in root path

    exp_idx = exp_idxs[0]
    data_path = root_path+'/{}/'.format(exp_idx)

    print(root_path,data_path)
    config_path = data_path+'config.yaml'
    with open(config_path) as f_tmp:
        config =  yaml.load(f_tmp, Loader=yaml.FullLoader)
    # camera_matrix = np.array(config['cam']['camera_matrix']['data']).reshape((3,3))
    # dist_coefs = np.array(config['cam']['dist_coeff']['data'])
    dist_coefs = None

    if side =='l':
        camera_matrix = np.array(config['cam']['RECT_M1']['data']).reshape((3,3))
        print("camera matrix", camera_matrix)
        # data_path =os.path.join(data_path,'rect_left')
    else:
        camera_matrix = np.array(config['cam']['RECT_M2']['data']).reshape((3,3))
        # data_path =os.path.join(data_path,'rect_right')

    pose_estimation.improve_segmentation(root_path,data_path,config['dataset'],camera_matrix,dist_coefs)

if __name__ == "__main__":
    main()

