# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np
from common.quaternion import qmul_np, qmul, qrot

class Skeleton:
    def __init__(self, offsets, parents, joints_left=None, joints_right=None):
        assert len(offsets) == len(parents)
        
        self._offsets = torch.FloatTensor(offsets)
        self._parents = np.array(parents)
        self._joints_left = joints_left
        self._joints_right = joints_right
        self._compute_metadata()
    
    def cuda(self):
        self._offsets = self._offsets.cuda()
        return self
    
    def num_joints(self):
        return self._offsets.shape[0]
    
    def offsets(self):
        return self._offsets
    
    def parents(self):
        return self._parents
    
    def has_children(self):
        return self._has_children
    
    def children(self):
        return self._children
    
    def remove_joints(self, joints_to_remove, propagate_offset=True):
        """
        Remove the joints specified in 'joints_to_remove' from the
        skeleton definition
        """
        kept_joints = [joint for joint in range(len(self._parents))
                      if joint not in joints_to_remove]

        # set parent to parents' parent if removed
        # down propagate their offset
        # does not work for h36m shoulders
        for i in range(len(self._parents)):
            while self._parents[i] in joints_to_remove:
                if propagate_offset:
                    self._offsets[i] += self._offsets[self._parents[i]]
                self._parents[i] = self._parents[self._parents[i]]      

        # recount indices for parent list
        index_offsets = np.zeros(len(self._parents), dtype=int)
        new_parents = []
        for i, parent in enumerate(self._parents):
            if i not in joints_to_remove:
                new_parents.append(parent - index_offsets[parent])
            else:
                index_offsets[i:] += 1
        self._parents = np.array(new_parents)

        # fix indices for joints_left & joints_right
        if self._joints_left is not None:
            new_joints_left = []
            for joint in self._joints_left:
                if joint in kept_joints:
                    new_joints_left.append(joint - index_offsets[joint])
            self._joints_left = new_joints_left
        if self._joints_right is not None:
            new_joints_right = []
            for joint in self._joints_right:
                if joint in kept_joints:
                    new_joints_right.append(joint - index_offsets[joint])
            self._joints_right = new_joints_right

        self._offsets = self._offsets[kept_joints]
        self._compute_metadata()

        return kept_joints

        
    def forward_kinematics(self, rotations, root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where N = batch size, L = sequence length, J = number of joints):
         -- rotations: (N, L, J, 4) tensor of unit quaternions describing the local rotations of each joint.
         -- root_positions: (N, L, 3) tensor describing the root joint positions.
        """
        assert len(rotations.shape) == 4
        assert rotations.shape[-1] == 4

        positions_world = []
        rotations_world = []

        expanded_offsets = self._offsets.expand(rotations.shape[0], rotations.shape[1],
                                                   self._offsets.shape[0], self._offsets.shape[1])

        # Parallelize along the batch and time dimensions
        for i in range(self._offsets.shape[0]):
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(rotations[:, :, 0])
            else:
                positions_world.append(qrot(rotations_world[self._parents[i]], expanded_offsets[:, :, i]) \
                                       + positions_world[self._parents[i]])
                if self._has_children[i]:
                    rotations_world.append(qmul(rotations_world[self._parents[i]], rotations[:, :, i]))
                else:
                    # This joint is a terminal node -> it would be useless to compute the transformation
                    rotations_world.append(None)

        return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2)
    
    def joints_left(self):
        return self._joints_left
    
    def joints_right(self):
        return self._joints_right

    def _all_children(self, parent):
      cs = []
      for c in self._children[parent]:
          cs.append(c)
          cs.extend(self._all_children(c))
      return cs

        
    def _compute_metadata(self):
        self._has_children = np.zeros(len(self._parents)).astype(bool)
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._has_children[parent] = True

        self._children = []
        for i, parent in enumerate(self._parents):
            self._children.append([])
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._children[parent].append(i)