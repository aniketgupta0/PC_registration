import logging
import os
import pickle

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.se3_numpy import se3_init, se3_transform, se3_inv
# from models.kpconv.backbone_kpconv.kpconv import batch_grid_subsampling_kpconv_gpu

import MinkowskiEngine as ME
import torch.nn.functional as F
from kiss_icp.pybind import kiss_icp_pybind

def voxel_down_sample(points: np.ndarray, voxel_size: float):
    _points = kiss_icp_pybind._Vector3dVector(points)
    return np.asarray(kiss_icp_pybind._voxel_down_sample(_points, voxel_size))


class KittiDataset(Dataset):
    def __init__(self, config, phase, transforms=None):
        super(KittiDataset, self).__init__()
        self.root = config.root
        self.phase = phase

        # Set downsampling parameters
        self.downsample = config.downsample
        self.alpha = config.alpha
        self.beta = config.beta

        self.initial_pose = np.eye(4)

        if self.phase=="train":
            self.sequences = ['{:02d}'.format(i) for i in range(1) if i!=config.validation_seq]
            # self.sequences = ["05"]
        elif self.phase=="val":
            self.sequences = [config.validation_seq]
        elif self.phase=="test":
            self.sequences = ['{:02d}'.format(i) for i in range(11, 22)]
        else:
            raise ValueError(f"Unknown mode{self.phase} (Correct modes: train, test, val)")

        self.poses_wrt_world = {} # Holds poses for all sequences wrt to world (cam0)
        self.poses_t2wt1 = {} # Holds poses for consecutive scans
        self.data_list = [] # Holds all the data with poses (if available)

        for seq in self.sequences:
            # 1: Get inbetween pose
            # 1.1: Read all the poses from the file
            if self.phase=="train" or self.phase=="val":
                print(f"This is the sequence: {seq}")
                pose_path = os.path.join(self.root, 'poses') + f"/{seq}.txt"
                self.poses_wrt_world[seq] = self._read_pose(pose_path)

            # 1.2; Read calibration file
            calib_path = os.path.join(self.root, "sequences", seq, "calib.txt")
            calib_dict = self.read_calib_file(calib_path)

            # 1.3: Update the relative poses
            self.get_relative_pose(calib_dict)

            # inbetween_poses = self.get_inbetween_poses(calib_dict)
            # print(f"sequences: {self.sequences}")
            # print(f"Len t2wt1: {len(self.poses_t2wt1['00'])}")
            # print(f"len inbetween_poses: {len(inbetween_poses)}")

            # a = np.array(self.poses_t2wt1['00'])
            # b = np.array(inbetween_poses)

            # print(f"Error: {np.sum(b-a)}")
            # assert(self.poses_t2wt1.shape == inbetween_poses.shape)


            # 2. Get the pc pairs
            # 2.1: Read all the pc's
            velo_path = os.path.join(self.root, 'sequences', seq, 'velodyne')
            for i, vf in enumerate(sorted(os.listdir(velo_path))):
                if i == 0 and vf.endswith('.bin'):
                    vf_path1 = os.path.join(self.root, 'sequences', seq, 'velodyne', vf)
                elif vf.endswith('.bin'):
                    vf_path2 = os.path.join(self.root, 'sequences', seq, 'velodyne', vf)
                    data = [vf_path1, vf_path2]
                    vf_path1 = vf_path2

                    if self.phase=="train" or self.phase=="val":
                        pose = self.poses_t2wt1[seq][i-1]
                        data.append(pose)

                    self.data_list.append(data)

    def get_inbetween_poses(self, calib_dict):
        inbetween_poses = []
        for seq in self.sequences:
            for i, pose in enumerate(self.poses_wrt_world[seq]):
                if i==0:
                    prev_pose = np.reshape(pose, (4,4))
                else:
                    curr_pose = np.reshape(pose, (4,4))
                    inbetween_poses.append(np.linalg.inv(curr_pose) @ prev_pose)

                    prev_pose = curr_pose
        
        return inbetween_poses

    def read_calib_file(self, filepath):
        """Read in a calibration file and parse into a dictionary."""
        data = {}

        with open(filepath, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        return data

    def get_relative_pose(self, calib_dict):
        """Ground truth poses in the Kitti dataset are given wrt to the world frame (cam0 here)
        This function converts those poses to poses between two consecutive scans"""
        for seq in self.sequences:
            self.poses_t2wt1[seq] = []
            for i, pose in enumerate(self.poses_wrt_world[seq]):
                if i==0:
                    prev_pose = np.reshape(pose, (4,4))
                else:
                    curr_pose = np.reshape(pose, (4,4))
                    self.poses_t2wt1[seq].append(np.linalg.inv(curr_pose) @ prev_pose)
                    
                    prev_pose = curr_pose

    def _pcread(self, path):
        """Read the pointcloud from the filepath"""
        frame_points = np.fromfile(path, dtype=np.float32)
        return frame_points.reshape((-1,4))[:, 0:3]

    def _read_pose(self, file_path):
        """Read the pose file from the filepath"""
        pose_list = []
        with open(file_path) as file:
            while True:
                line = file.readline()
                if not line:
                    break
                T = np.fromstring(line, dtype=np.float32, sep=' ')
                T = np.append(T, [0,0,0,1])

                pose_list.append(T)
        return pose_list

    def __getitem__(self, index):
        if self.phase=="train" or self.phase=="val":
            pc1, pc2, pose = self.data_list[index]
        else:
            pc1, pc2 = self.data_list[index]
            pose = None

        data = {}
        
        pc1 = self._pcread(pc1)
        pc2 = self._pcread(pc2)
        
        if self.downsample:
            # Voxel downsampling 1
            pc1 = voxel_down_sample(pc1, 0.5)
            pc2 = voxel_down_sample(pc2, 0.5)

            # Voxel downsampling 2
            pc1 = voxel_down_sample(pc1, 1.5)
            pc2 = voxel_down_sample(pc2, 1.5)

        data['src_xyz'] = torch.from_numpy(pc1.astype(np.float32))
        data['tgt_xyz'] = torch.from_numpy(pc2.astype(np.float32))

        data['pose'] = torch.from_numpy(pose[:3, :].astype('float32'))

        return data

    def __len__(self):
        return len(self.data_list)