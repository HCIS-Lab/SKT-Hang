import os, glob, json, sys
sys.path.append('../')

import numpy as np
from tqdm import tqdm

from scipy.spatial.transform import Rotation as R
from pointnet2_ops.pointnet2_utils import furthest_point_sample
import torch
from torch.utils.data import Dataset
# from torchvision import datasets
# from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from utils.training_utils import get_model_module, optimizer_to_device, normalize_pc

# class KptrajReconAffordanceRot3dDataset(Dataset):
#     def __init__(self, dataset_dir, num_steps=30, wpt_dim=6, sample_num_points=1000, enable_traj=True, enable_affordance=False, enable_partseg=False):
        
#         assert os.path.exists(dataset_dir), f'{dataset_dir} not exists'
#         assert wpt_dim == 6 or wpt_dim == 3, f'wpt_dim should be 3 or 6'

#         dataset_subdirs = glob.glob(f'{dataset_dir}/*')
#         self.enable_affordance = enable_affordance
#         self.enable_traj = enable_traj
#         self.enable_partseg = enable_partseg

#         self.type = "residual" if "residual" in dataset_dir else "absolute"
#         self.traj_len = num_steps
#         self.wpt_dim = wpt_dim
#         self.sample_num_points = sample_num_points

#         self.shape_list = [] # torch tensor
#         self.center_list = []
#         self.scale_list = [] 
#         self.partseg_list = []
#         self.affordance_list = []
#         self.traj_list = []
#         for dataset_subdir in dataset_subdirs:

#             shape_files = glob.glob(f'{dataset_subdir}/affordance*.npy') # point cloud with affordance score (Nx4), the first element is the contact point
#             shape_list_tmp = []
#             affordance_list_tmp = []
#             partseg_list_tmp = []
#             center_list_tmp = []
#             scale_list_tmp = [] 
#             for shape_file in shape_files:
#                 pcd = np.load(shape_file).astype(np.float32)
#                 points = pcd[:,:3]

#                 centroid_points, center, scale = normalize_pc(points, copy_pts=True) # points will be in a unit sphere
#                 centroid_points = torch.from_numpy(centroid_points).unsqueeze(0).to(device).contiguous()
#                 input_pcid = furthest_point_sample(centroid_points, self.sample_num_points).long().reshape(-1)  # BN
#                 centroid_points = centroid_points[0, input_pcid, :].squeeze()

#                 center = torch.from_numpy(center).to(device)

#                 shape_list_tmp.append(centroid_points)
#                 center_list_tmp.append(center)
#                 scale_list_tmp.append(scale)

#                 if enable_affordance == True:
#                     affordance = pcd[:,3]
#                     affordance = torch.from_numpy(affordance).to(device)
#                     affordance = affordance[input_pcid]
#                     affordance_list_tmp.append(affordance)
            
#                 if enable_partseg == True:
#                     assert  pcd.shape[1] > 4, f'your affordance map does not contain part segmentation information'
#                     partseg = pcd[:,4]
#                     partseg = torch.from_numpy(partseg).to(device)
#                     partseg = partseg[input_pcid]
#                     partseg_list_tmp.append(partseg)

#             self.shape_list.append(shape_list_tmp)
#             self.center_list.append(center_list_tmp)
#             self.scale_list.append(scale_list_tmp)
#             if enable_affordance == True:
#                 self.affordance_list.append(affordance_list_tmp)
#             if enable_affordance == True:
#                 self.partseg_list.append(partseg_list_tmp)

#             traj_list_tmp = []
#             if enable_traj: 
#                 traj_files = glob.glob(f'{dataset_subdir}/*.json') # trajectory in 6d format
#                 for traj_file in traj_files:
                    
#                     f_traj = open(traj_file, 'r')
#                     traj_dict = json.load(f_traj)
#                     waypoints = np.asarray(traj_dict['trajectory'])

#                     if self.type == "residual":
#                         waypoints[0, :3] = 0 # clear position of the first waypoint because this position will be given by affordance network

#                     waypoints = torch.FloatTensor(waypoints).to(device)

#                     traj_list_tmp.append(waypoints)

#                 self.traj_list.append(traj_list_tmp)

#         if enable_traj and enable_partseg and enable_affordance: # 111
#             assert len(self.shape_list) == len(self.traj_list) == len(self.partseg_list) == len(self.affordance_list), 'inconsistent length of shapes and partseg and trajectories and affordance'
#         elif enable_traj and enable_partseg: # 101
#             assert len(self.shape_list) == len(self.traj_list) == len(self.partseg_list), 'inconsistent length of shapes and partseg and trajectories'
#         elif enable_traj and enable_affordance: # 110
#             assert len(self.shape_list) == len(self.traj_list) == len(self.affordance_list), 'inconsistent length of shapes and affordance and trajectories'
#         elif enable_partseg and enable_affordance: # 011
#             assert len(self.shape_list) == len(self.traj_list) == len(self.affordance_list), 'inconsistent length of shapes and affordance and trajectories'
#         elif enable_partseg: # 001
#             assert len(self.shape_list) == len(self.partseg_list), 'inconsistent length of shapes and partseg'
#         elif enable_affordance: # 010
#             assert len(self.shape_list) == len(self.affordance_list), 'inconsistent length of shapes and affordance'
#         elif enable_traj: # 100
#             assert len(self.shape_list) == len(self.traj_list), 'inconsistent length of shapes and trajectories'

#         self.size = len(self.shape_list)

#     def print_data_shape(self):
#         print(f'sample_num_points : {self.sample_num_points}')
#         print(f"shape : {len(self.shape_list)}")
#         if self.enable_affordance:
#             print(f"affordance : {len(self.affordance_list)}")
#         if self.enable_traj:
#             print(f"trajectory : {len(self.traj_list)}")
        
#     def __len__(self):
#         return self.size
    
#     def __getitem__(self, index):

#         # for point cloud processing
#         num_pcd = len(self.shape_list[index])
#         shape_id = np.random.randint(0, num_pcd)
#         points = self.shape_list[index][shape_id]
#         center = self.center_list[index][shape_id]
#         scale = self.scale_list[index][shape_id]

#         # for affordance processing if enabled
#         affordance = None
#         if self.enable_affordance:
#             affordance = self.affordance_list[index][shape_id]

#         # for partseg processing if enabled
#         partseg = None
#         if self.enable_partseg:
#             partseg = self.partseg_list[index][shape_id]
            
#         # for waypoint preprocessing
#         waypoints = None
#         if self.enable_traj:
#             num_traj = len(self.traj_list[index])
#             traj_id = np.random.randint(0, num_traj)
#             wpts = self.traj_list[index][traj_id]
#             waypoints = wpts.clone()
#             if self.type == "absolute":
#                 waypoints[:,:3] = (waypoints[:,:3] - center) / scale
#             elif self.type == "residual":
#                 waypoints[1:,:3] = waypoints[1:,:3] / scale
#             else :
#                 print(f"dataset type undefined : {self.type}")
#                 exit(-1)

#         # ret value
#         if self.enable_traj and self.enable_affordance and self.enable_partseg: # 111
#             return points, affordance, partseg, waypoints[:,:self.wpt_dim]
#         elif self.enable_traj and self.enable_partseg: # 101
#             return points, partseg, waypoints[:,:self.wpt_dim]
#         elif self.enable_affordance and self.enable_traj:
#             return points, affordance, waypoints[:,:self.wpt_dim]
#         elif self.enable_affordance and self.enable_partseg:
#             return points, affordance, partseg
#         elif self.enable_partseg:
#             return points, partseg
#         elif self.enable_affordance:
#             return points, affordance
#         elif self.enable_traj:
#             return points, waypoints[:,:self.wpt_dim]
#         return points
    
class KptrajReconAffordanceDataset(Dataset):
    def __init__(self, dataset_dir, num_steps=30, wpt_dim=6, sample_num_points=1000, enable_traj=True, affordance_type=0, device='cuda', with_noise=False):
        
        assert os.path.exists(dataset_dir), f'{dataset_dir} not exists'
        assert wpt_dim == 9  or wpt_dim == 6 or wpt_dim == 3, f'wpt_dim should be 3 or 6'
        
        self.with_noise = with_noise
        self.noise_pos_scale = 0.0002 # unit: meter
        self.noise_rot_scale = 0.5 * torch.pi / 180 # unit: meter

        self.device = device
        self.affordance_type = affordance_type
        self.affordance_name = {
            0:'none',
            1:'affordance',
            2:'partseg',
            3:'both',
            4:'fusion',
        }[affordance_type]

        dataset_subdirs = glob.glob(f'{dataset_dir}/*')
        self.enable_traj = enable_traj

        self.type = "residual" if "residual" in dataset_dir else "absolute"
        self.traj_len = num_steps
        self.wpt_dim = wpt_dim
        self.sample_num_points = sample_num_points
        
        self.shape_list = [] # torch tensor
        self.center_list = []
        self.scale_list = [] 
        self.affordance_list = []
        self.partseg_list = []
        self.fusion_list = []
        self.traj_list = []
        for i, dataset_subdir in enumerate(tqdm(dataset_subdirs)):

            shape_files = glob.glob(f'{dataset_subdir}/affordance*.npy') # point cloud with affordance score (Nx4), the first element is the contact point
            shape_list_tmp = []
            affordance_list_tmp = []
            fusion_list_tmp = []
            partseg_list_tmp = []
            center_list_tmp = []
            scale_list_tmp = [] 
            pcd_cps = None
            for shape_file in shape_files:
                pcd = np.load(shape_file).astype(np.float32)
                points = pcd[:,:3]
                pcd_cp = pcd[0, :3]
                pcd_cps = pcd_cp.reshape(1, 3) if pcd_cps is None else np.vstack((pcd_cps, pcd_cp)) 

                centroid_points, center, scale = normalize_pc(points, copy_pts=True) # points will be in a unit sphere
                centroid_points = torch.from_numpy(centroid_points).unsqueeze(0).to(device).contiguous()
                
                input_pcid = None
                point_num = centroid_points.shape[1]
                if point_num >= self.sample_num_points:
                    input_pcid = furthest_point_sample(centroid_points, self.sample_num_points).long().reshape(-1)  # BN
                    centroid_points = centroid_points[0, input_pcid, :].squeeze()
                else :
                    mod_num = self.sample_num_points % point_num
                    repeat_num = int(self.sample_num_points // point_num)
                    input_pcid = furthest_point_sample(centroid_points, mod_num).long().reshape(-1)  # BN
                    centroid_points = torch.cat([centroid_points.repeat(1, repeat_num, 1), centroid_points[:, input_pcid]], dim=1).squeeze()
                    input_pcid = torch.cat([torch.arange(0, point_num).int().repeat(repeat_num).to(self.device), input_pcid])
                
                center = torch.from_numpy(center).to(device)

                shape_list_tmp.append(centroid_points)
                center_list_tmp.append(center)
                scale_list_tmp.append(scale)

                if self.affordance_name == 'affordance' or self.affordance_name == 'both':
                    affordance = pcd[:,3]
                    affordance = torch.from_numpy(affordance).to(device)
                    affordance = affordance[input_pcid]
                    affordance_list_tmp.append(affordance)

                if self.affordance_name == 'partseg' or self.affordance_name == 'both':
                    assert  pcd.shape[1] > 4, f'your affordance map does not contain part segmentation information'
                    partseg = pcd[:,4]
                    partseg = torch.from_numpy(partseg).to(device)
                    partseg = partseg[input_pcid]
                    partseg_list_tmp.append(partseg)

                if self.affordance_name == 'fusion':
                    assert  pcd.shape[1] > 4, f''
                    affordance = pcd[:,3]
                    partseg = pcd[:,4]
                    fusion = (pcd[:,3] + pcd[:,4]) / 2 # just average it
                    fusion = torch.from_numpy(fusion).to(device)
                    fusion = fusion[input_pcid]
                    fusion_list_tmp.append(fusion)
            
            self.shape_list.append(shape_list_tmp)
            self.center_list.append(center_list_tmp)
            self.scale_list.append(scale_list_tmp)
            if self.affordance_name == 'affordance' or self.affordance_name == 'both':
                self.affordance_list.append(affordance_list_tmp)
            if self.affordance_name == 'partseg' or self.affordance_name == 'both':
                self.partseg_list.append(partseg_list_tmp)
            if self.affordance_name == 'fusion':
                self.fusion_list.append(fusion_list_tmp)

            traj_list_tmp = []
            if enable_traj: 
                traj_files = glob.glob(f'{dataset_subdir}/*.json')[:1] # trajectory in 7d format
                    
                for traj_file in traj_files:
                    
                    f_traj = open(traj_file, 'r')
                    traj_dict = json.load(f_traj)

                    waypoints = np.asarray(traj_dict['trajectory'])
                        
                    if self.wpt_dim == 3:
                        waypoints = waypoints[:, :3]
                        
                    if self.type == "residual" and self.wpt_dim == 6:
                        first_rot_matrix = R.from_rotvec(waypoints[0, 3:]).as_matrix() # omit absolute position of the first waypoint
                        first_rot_matrix_xy = (first_rot_matrix.T).reshape(-1)[:6] # the first, second column of the rotation matrix
                        waypoints[0] = first_rot_matrix_xy # rotation only (6d rotation representation)

                    if self.wpt_dim == 9:
                        waypoints_9d = np.zeros((waypoints.shape[0], 9))
                        waypoints_9d[:, :3] = waypoints[:, :3]
                        rot_matrix = R.from_rotvec(waypoints[:, 3:]).as_matrix() # omit absolute position of the first waypoint
                        rot_matrix_xy = np.transpose(rot_matrix, (0, 2, 1)).reshape((waypoints.shape[0], -1))[:, :6] # the first, second column of the rotation matrix
                        waypoints_9d[:, 3:] = rot_matrix_xy # rotation only (6d rotation representation)
                        waypoints = waypoints_9d

                    waypoints = torch.FloatTensor(waypoints).to(device)
                    traj_list_tmp.append(waypoints)

                self.traj_list.append(traj_list_tmp)

        if enable_traj and self.affordance_name == 'both': 
            assert len(self.shape_list) == len(self.traj_list) == len(self.partseg_list) == len(self.affordance_list), 'inconsistent length of shapes and partseg and trajectories and affordance'
        elif enable_traj and self.affordance_name == "partseg":
            assert len(self.shape_list) == len(self.traj_list) == len(self.partseg_list), 'inconsistent length of shapes and partseg and trajectories'
        elif enable_traj and self.affordance_name == "affordance":
            assert len(self.shape_list) == len(self.traj_list) == len(self.affordance_list), 'inconsistent length of shapes and affordance and trajectories'
        elif enable_traj and self.affordance_name == "fusion":
            assert len(self.shape_list) == len(self.traj_list) == len(self.fusion_list), 'inconsistent length of shapes and affordance and trajectories'
        elif self.affordance_name == 'both':
            assert len(self.shape_list) == len(self.traj_list) == len(self.affordance_list) == len(self.partseg_list), 'inconsistent length of shapes and affordance and partseg and trajectories'
        elif self.affordance_name == 'partseg':
            assert len(self.shape_list) == len(self.partseg_list), 'inconsistent length of shapes and partseg'
        elif self.affordance_name == "affordance":
            assert len(self.shape_list) == len(self.affordance_list), 'inconsistent length of shapes and affordance'
        elif self.affordance_name == "fusion":
            assert len(self.shape_list) == len(self.fusion_list), 'inconsistent length of shapes and affordance'
        elif enable_traj: # 100
            assert len(self.shape_list) == len(self.traj_list), 'inconsistent length of shapes and trajectories'
        
        self.size = len(self.shape_list)

    def print_data_shape(self):
        print(f'dataset size : {self.size}')
        print(f'sample_num_points : {self.sample_num_points}')
        if self.enable_traj:
            print(f"trajectory : {len(self.traj_list)}")
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):

        # for point cloud processing
        num_pcd = len(self.shape_list[index])
        shape_id = np.random.randint(0, num_pcd)
        points = self.shape_list[index][shape_id].clone()
        center = self.center_list[index][shape_id].clone()
        scale = self.scale_list[index][shape_id]

        # noise to point cloud
        if self.with_noise:
            point_noises = torch.randn(points[1:, :3].shape).to(self.device) * self.noise_pos_scale / scale
            points[1:, :3] += point_noises

        # for affordance processing if enabled
        affordance = None
        if self.affordance_name == 'affordance' or self.affordance_name == 'both':
            affordance = self.affordance_list[index][shape_id]

        # for partseg processing if enabled
        partseg = None
        if self.affordance_name == 'partseg' or self.affordance_name == 'both':
            partseg = self.partseg_list[index][shape_id]

        # for fusion processing if enabled
        fusion = None
        if self.affordance_name == 'fusion':
            fusion = self.fusion_list[index][shape_id]

        # for waypoint preprocessing
        waypoints = None
        if self.enable_traj:
            num_traj = len(self.traj_list[index])
            traj_id = np.random.randint(0, num_traj)
            wpts = self.traj_list[index][traj_id]
            waypoints = wpts.clone()
            
            # noise to waypoints
            if self.with_noise:
                pos_noises = torch.randn(waypoints[1:, :3].shape).to(self.device) * self.noise_pos_scale * 0.1
                waypoints[1:, :3] += pos_noises
                if self.wpt_dim > 3:
                    rot_noises = torch.randn(waypoints[1:, 3:].shape).to(self.device) * self.noise_rot_scale * 0.1
                    waypoints[1:, 3:] += rot_noises

            if self.type == "absolute":
                waypoints[:, :3] = (waypoints[:, :3] - center) / scale
            elif self.type == "residual":
                waypoints[1:, :3] = waypoints[1:, :3] / scale
            else :
                print(f"dataset type undefined : {self.type}")
                exit(-1)

            contact_point = points[0, :3]
            first_wpt = None
            if self.wpt_dim == 9 or (self.type == "absolute" and self.wpt_dim == 6):
                first_wpt = torch.hstack((contact_point, waypoints[0, 3:])) # use the second rot as the first rot
            elif self.wpt_dim == 3:
                first_wpt = contact_point

            if (self.wpt_dim == 9 or self.wpt_dim == 3 or (self.type == "absolute" and self.wpt_dim == 6)):
                if torch.sum(torch.abs(first_wpt[:3] - waypoints[0, :3])) > 0.2: # magic number
                    waypoints = torch.vstack([first_wpt, waypoints])
                else :
                    waypoints[0] = first_wpt
                waypoints = waypoints[:self.traj_len]

        # ret value
        if self.enable_traj and self.affordance_name == 'both': 
            return points, affordance, partseg, waypoints[:,:self.wpt_dim]
        elif self.enable_traj and self.affordance_name == 'affordance':
            return points, affordance, waypoints[:,:self.wpt_dim]
        elif self.enable_traj and self.affordance_name == 'partseg':
            return points, partseg, waypoints[:,:self.wpt_dim]
        elif self.enable_traj and self.affordance_name == 'fusion':
            return points, fusion, waypoints[:,:self.wpt_dim]
        elif self.enable_traj:
            return points, waypoints[:,:self.wpt_dim]
        elif self.affordance_name == 'affordance':
            return points, affordance
        elif self.affordance_name == 'partseg':
            return points, partseg
        elif self.affordance_name == 'both':
            return points, affordance, partseg
        elif self.affordance_name == 'fusion':
            return points, fusion
        return points

if __name__=="__main__":
    
    dataset_dir = "../../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000/train"
    dataset = KptrajReconAffordanceDataset(dataset_dir)
    dataset.print_data_shape()

