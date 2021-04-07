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
      if 'Position' in feature:
        cdf_content /= 1000 # Meters instead of millimeters
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
            #r = (r + 180) % 360 - 180

            # change order from zxy to xyz
            r = np.roll(r, 2, axis=-1)

            
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
            # change from euler to quaternion representation
            r = qfix(euler_to_quaternion(r, 'zxy'))
            
            rot_3d[subject][action] = r
            traj[subject][action] = t / 1000. # convert to meter
    
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

def load_h36m(path, keep_feet=True, keep_shoulders=True, downsample=1):
    features_file = Path(path) / 'h36m_features.npz'

    # convert from cdf files if necessary
    if not features_file.exists():
        convert_h36m_from_cdf(path)

    print('Load H36m data...')
    dataset = Human36mDataset(features_file.resolve(), keep_feet, keep_shoulders)

    if downsample > 1:
        dataset.downsample(downsample)

    print('Computing ground-truth 2D poses...')
    dataset.calc_2d_pos(normalized=True) # use normalized camera frames
    print('Done.')

    return dataset

def create_meta(dataset):
    metadata = {
        'num_joints': dataset.skeleton().num_joints(),
        'keypoints_symmetry': [dataset.skeleton().joints_left(), 
                              dataset.skeleton().joints_right()]
    }
    return metadata

# prepares dataset and saves preprocessed npz file
def create_h36m_gt(path, keep_feet=True, keep_shoulders=True, downsample=1):
    dataset = load_h36m(path, keep_feet, keep_shoulders, downsample)

    metadata = create_meta(dataset)
    
    #extract relevant features (2d keypoints & rotations)
    pose = {}
    for subject in dataset.subjects():
        pose[subject] = {}
        for action in dataset[subject]:
            pose_data = {}
            pose_data['rotations'] = dataset[subject][action]['rotations']
            pose_data['positions_2d'] = dataset[subject][action]['positions_2d']
            pose[subject][action] = pose_data

    print('Saving 2d data...')
    file_2d = Path(path) / 'h36m_2d.npz'
    np.savez_compressed(file_2d.resolve(), pose=pose, metadata=metadata)
    print('Saving complete.')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Dataset preparation')
  parser.add_argument('-p', '--path', default='./data', type=str, metavar='NAME', help='dataset path')
  parser.add_argument('--no-feet', action='store_false', dest='include_feet',
                        help='disable the inclusion of feet in the model')
  parser.set_defaults(include_feet=False)

  args = parser.parse_args()

  create_h36m_gt(args.path, not args.include_feet)





