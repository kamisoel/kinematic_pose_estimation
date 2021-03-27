import cdflib
import numpy as np
from pathlib import Path
import argparse

from common.dataset_h36m import Human36mDataset
from common.camera import *
from common.quaternion import *


def load_h36m_feature(path, feature='D3_Positions', n_joints=32,
                      actions=None, subjects=None):
  values = {}
  if subjects is None:
    subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']

  for subject in subjects:
    values[subject] = {}

    feature_path = Path(path) / subject / 'MyPoseFeatures' / feature
    
    for f in feature_path.glob('*.cdf'):
      action = f.stem
      if actions is not None and action not in actions:
        continue

      if subject == 'S11' and action == 'Directions':
        continue # Discard corrupted video

      cdf_content = cdflib.CDF(f.resolve())['Pose']
      #if 'Position' in feature:
      #  cdf_content /= 1000 # Meters instead of millimeters
      #if 'Angle' in feature:
      #  cdf_content = cdf_content[:, :, 3:] # drop hip position

      n_dim = cdf_content.shape[-1] // n_joints
      values[subject][action] = cdf_content.reshape(-1, n_joints, n_dim).astype('float32')
  
  return values


def preprocess_rotations(data):
    rot_3d = {}
    traj = {}
    for subject, actions in data.items():
        rot_3d[subject] = {}
        traj[subject] = {}
        for action in actions:
            r = data[subject][action][:,1:,:]
            t = data[subject][action][:,0,:]

            # keep angles in [-180°,180°]
            r = (r + 180) % 360 - 180
            # change order from zyx to xyz
            #r = np.flip(r, axis=-1)
            
            # add zero rotation for end-effectors if necessary
            if r.shape[1] == 25:
                l = len(r)
                r = np.concatenate((
                      r[:,:5,:], np.zeros((l,1,3)), r[:,5:9,:], np.zeros((l,1,3)),
                      r[:,9:13,:], np.zeros((l,1,3)), r[:,13:19,:], 
                      np.zeros((l,2,3)), r[:,19:,:], np.zeros((l,2,3))),
                    axis=1)
            
            # degrees to radians
            r = np.deg2rad(r)
            #change from euler to quaternion representation
            r = euler_to_quaternion(r, 'zyx')
            
            rot_3d[subject][action] = r
            traj[subject][action] = t
    
    return rot_3d, traj


def convert_h36m_from_cdf(path):
  print('Load cdf files...')
  #pos_2d = load_h36m_feature(input_dir, 'D2_Positions') #will be calculated from 3d
  pos_3d = load_h36m_feature(path, 'D3_Positions')
  angles = load_h36m_feature(path, 'D3_Angles', n_joints=26) # 25 rotation + 1 for trajectory
  rot_3d, traj = preprocess_rotations(angles) # return rotation for 32 joints (inc. end-efectors) + trajectory

  
  print('Save in npz format...')
  out_file = Path(path) / 'h36m_features.npz'
  np.savez_compressed(out_file.resolve(), positions_3d=pos_3d, 
                      rotations_3d=rot_3d, trajectory=traj)
  
  print('Done.')


def prepare_2d_data(path, remove_feet=False):
  features_file = Path(path) / 'h36m_features.npz'

  if not features_file.exists():
    convert_h36m_from_cdf(path)

  print('Preprocess data and calculate 2D positions...')
  dataset = Human36mDataset(features_file.resolve(), remove_feet)

  metadata = {
      'num_joints': dataset.skeleton().num_joints(),
      'keypoints_symmetry': [dataset.skeleton().joints_left(), 
                             dataset.skeleton().joints_right()]
  }

  print('Saving 2d data...')
  file_2d = Path(path) / 'h36m_gt.npz'
  np.savez_compressed(file_2d.resolve(), pose=dataset._data, metadata=metadata)
  print('Done.')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Dataset preparation')
  parser.add_argument('-p', '--path', default='./data', type=str, metavar='NAME', help='dataset path')
  parser.add_argument('--no-feet', action='store_false', dest='include_feet',
                        help='disable the inclusion of feet in the model')
  parser.set_defaults(include_feet=False)

  args = parser.parse_args()

  prepare_2d_data(args.path, not args.include_feet)





