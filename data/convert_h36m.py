import cdflib
import numpy as np
from pathlib import Path


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
    new_rot_3d = {}
    for subject, actions in data.items():
        new_rot_3d[subject] = {}
        for action in actions:
            r = data[subject][action]
            # keep angles in [-180°,180°]
            r = (r + 180) % 360 - 180
            # change order from zyx to xyz
            #r = np.flip(r, axis=-1)
            
            # add zero rotation for end-effectors if necessary
            if r.shape[1] == 26:
                l = len(r)
                r = np.concatenate((
                      r[:,:6,:], np.zeros((l,1,3)), r[:,6:10,:], np.zeros((l,1,3)),
                      r[:,10:14,:], np.zeros((l,1,3)), r[:,14:20,:], 
                      np.zeros((l,2,3)), r[:,20:,:], np.zeros((l,2,3))),
                    axis=1)

            #change from euler to quaternion representation
            r = euler_to_quaternion(r, 'zyx')
            
            new_rot_3d[subject][action] = r
    return new_rot_3d


def convert_h36m_from_cdf(path):
  print('Load cdf files...')
  #pos_2d = load_h36m_feature(input_dir, 'D2_Positions') #will be calculated from 3d
  pos_3d = load_h36m_feature(path, 'D3_Positions')
  rot_3d = load_h36m_feature(path, 'D3_Angles', n_joints=26) # 25 rotation + 1 for trajectory
  rot_3d = preprocess_rotations(rot_3d) # return rotation for 32 joints (inc. end-efectors) + trajectory

  
  print('Save in npz format...')
  out_file = Path(path) / 'h36m_features.npz'
  np.savez_compressed(out_file.resolve(), positions_3d=pos_3d, rotations_3d=rot_3d)
  
  print('Done.')