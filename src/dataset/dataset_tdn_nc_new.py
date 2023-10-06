import os, glob, json, sys, random
sys.path.append('../')

import numpy as np
from tqdm import tqdm

from scipy.spatial.transform import Rotation as R
from pointnet2_ops.pointnet2_utils import furthest_point_sample
import torch
from torch.utils.data import Dataset

from utils.training_utils import normalize_pc

class KptrajDeformAffordanceDataset(Dataset):
    def __init__(self, dataset_dir, category_file, num_steps=40, wpt_dim=6, sample_num_points=1000, device='cuda', with_noise=False, gt_trajs=1, mask=0):
        
        assert os.path.exists(dataset_dir), f'{dataset_dir} not exists'
        assert os.path.exists(category_file), f'{category_file} not exists'
        assert wpt_dim == 9  or wpt_dim == 6 or wpt_dim == 3, f'wpt_dim should be 3 or 6 or 9'
        
        self.with_noise = with_noise
        self.gt_trajs = gt_trajs
        self.noise_pos_scale = 0.0002 # unit: meter
        self.noise_rot_scale = 0.5 * torch.pi / 180 # unit: meter

        self.device = device
        
        # ================== read category information ================== #

        f_category_file = open(category_file, 'r')
        category_raw_data = f_category_file.readlines()
        category_data_dict = {}
        oepn = False
        for line in category_raw_data:
            if '==============================================================================================' in  line:
                oepn = not oepn
                if not oepn:
                    continue

                line_strip = line.strip()
                hook_name = line_strip.split(' => ')[0]
                hook_category = int(line_strip.split(' => ')[1])

                assert hook_name not in category_data_dict.keys()
                
                category_data_dict[hook_name] = {
                    'category': hook_category,
                    'center': 0
                }

            if 'center of class' in line:
                line_strip = line.strip()
                hook_name = line_strip.split(': ')[-1].strip()  # ex: center of class [0]: Hook_my_bar_easy => Hook_my_bar_easy
                hook_category = int(line_strip.split('[')[1].split(']')[0].strip()) # ex: center of class [0]: Hook_my_bar_easy => 0

                assert hook_name in category_data_dict.keys()
                assert category_data_dict[hook_name]['category'] == hook_category
                
                category_data_dict[hook_name]['center'] = 1

        # =============================================================== #

        # ================== config template trajectories ================== #

        dataset_subdirs = glob.glob(f'{dataset_dir}/*')
        template_hooks = []
        for dataset_subdir in dataset_subdirs:
            hook_name = dataset_subdir.split('/')[-1]
            if category_data_dict[hook_name]['center'] == 1:
                template_hooks.append(hook_name)
        
        # randomly drop n templates
        # legal_categories = random.sample(range(len(template_hooks)), len(template_hooks))
        legal_categories = range(len(template_hooks))[:(len(template_hooks)-mask)]
        
        for template_hook in template_hooks:
            item = f'{dataset_dir}/{template_hook}'
            old_index = dataset_subdirs.index(item)
            dataset_subdirs.insert(0, dataset_subdirs.pop(old_index))
        
        # self.template_dict = { i: {} for i in range(len(template_hooks))}
        self.template_dict = { i: {} for i in range(len(legal_categories))}

        # ================================================================== #

        self.type = "residual" if "residual" in dataset_dir else "absolute"
        self.traj_len = num_steps
        self.wpt_dim = wpt_dim
        self.sample_num_points = sample_num_points
        
        self.shape_list = [] 
        self.center_list = []
        self.scale_list = [] 
        self.fusion_list = []
        self.traj_list = []
        self.category_list = []

        for dataset_subdir in tqdm(dataset_subdirs):
            
            hook_name = dataset_subdir.split('/')[-1]
            current_type = 'template' if hook_name in template_hooks else 'normal'
            category = category_data_dict[hook_name]['category']

            if category not in legal_categories:
                continue

            shape_files = glob.glob(f'{dataset_subdir}/affordance*.npy') # point cloud with affordance score (Nx4), the first element is the contact point
            shape_list_tmp = []
            fusion_list_tmp = []
            center_list_tmp = []
            scale_list_tmp = [] 
            
            if current_type == 'normal':
                self.category_list.append(category)
            else:
                template_info = {
                    'shape': [],
                    'center': [],
                    'scale': [],
                    'fusion': [],
                    'traj': [],
                }
            
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

                assert  pcd.shape[1] > 4, f''
                fusion = (pcd[:,3] + pcd[:,4]) / 2 # just average it
                fusion = torch.from_numpy(fusion).to(device)
                fusion = fusion[input_pcid]
                fusion_list_tmp.append(fusion)
            
            if current_type == 'normal':
                self.shape_list.append(shape_list_tmp)
                self.center_list.append(center_list_tmp)
                self.scale_list.append(scale_list_tmp)
                self.fusion_list.append(fusion_list_tmp)
            else :
                template_info['shape'] = shape_list_tmp
                template_info['center'] = center_list_tmp
                template_info['scale'] = scale_list_tmp
                template_info['fusion'] = fusion_list_tmp

            traj_list_tmp = []
            traj_files = glob.glob(f'{dataset_subdir}/*.json') # trajectory in 7d format
                
            for traj_file in traj_files:
                
                f_traj = open(traj_file, 'r')
                traj_dict = json.load(f_traj)

                waypoints = np.asarray(traj_dict['trajectory'])

                if self.wpt_dim == 3:
                    waypoints = waypoints[:, :3]

                if self.wpt_dim == 6 and self.type == "residual":
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

            if current_type == 'normal':
                self.traj_list.append(traj_list_tmp)
            else :
                template_info['traj'] = traj_list_tmp
                self.template_dict[category] = template_info

        assert len(self.shape_list) == len(self.center_list) == len(self.scale_list) == len(self.fusion_list) == len(self.traj_list) == len(self.category_list), 'inconsistent length of shapes and affordance'
        self.size = len(self.shape_list)

    def print_data_shape(self):
        print(f'dataset size : {self.size}')
        print(f"trajectory : {len(self.traj_list)}")
        print(f'sample_num_points : {self.sample_num_points}')

    def set_index(self, val):
        self.index = val
        print(f'self.index = {self.index}')
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):

        ############################
        # for template information #
        ############################

        category = self.category_list[index]

        template_info = self.template_dict[category]
        temp_shape_id = np.random.randint(0, len(template_info['shape']))
        temp_traj_id = np.random.randint(0, len(template_info['traj']))

        temp_wpts = template_info['traj'][temp_traj_id].clone()
        temp_center = template_info['center'][temp_shape_id].clone()
        temp_scale = template_info['scale'][temp_shape_id]

        # noise to waypoints
        if self.with_noise:
            pos_noises = torch.randn(temp_wpts[1:, :3].shape).to(self.device) * self.noise_pos_scale 
            temp_wpts[1:, :3] += pos_noises * 0.1
            if self.wpt_dim > 3:
                rot_noises = torch.randn(temp_wpts[1:, 3:].shape).to(self.device) * self.noise_rot_scale 
                temp_wpts[1:, 3:] += rot_noises * 0.1

        if self.type == "absolute":
            temp_wpts[:, :3] = (temp_wpts[:, :3] - temp_center) / temp_scale
        elif self.type == "residual":
            temp_wpts[0 , :3] = (temp_wpts[ 0, :3] - temp_center) / temp_scale
            temp_wpts[1:, :3] =  temp_wpts[1:, :3] / temp_scale
        else :
            print(f"dataset type undefined : {self.type}")
            exit(-1)

        ###############################
        # for point cloud information #
        ###############################
        
        num_pcd = len(self.shape_list[index])
        shape_id = np.random.randint(0, num_pcd)
        points = self.shape_list[index][shape_id].clone()
        center = self.center_list[index][shape_id].clone()
        scale = self.scale_list[index][shape_id]

        # noise to point cloud
        if self.with_noise:
            point_noises = torch.randn(points[1:, :3].shape).to(self.device) * self.noise_pos_scale / scale
            points[1:, :3] += point_noises

        # for fusion processing if enabled
        fusion = self.fusion_list[index][shape_id]

        # for waypoint preprocessing
        num_traj = len(self.traj_list[index])
        traj_inds = random.sample(range(num_traj), self.gt_trajs)
        wpts_src = torch.stack(self.traj_list[index])[traj_inds]
        wpts = wpts_src.clone()

        # start = np.random.randint(0, num_traj - self.gt_trajs)
        # end = start + self.gt_trajs
        # wpts_src = self.traj_list[index][start:end]
        # wpts = torch.stack(wpts_src).clone()

        if self.type == "absolute":
            wpts[:, :, :3] = (wpts[:, :, :3] - center.unsqueeze(0)) / scale
        elif self.type == "residual":
            wpts[:, 0, :3] = (wpts[:, 0, :3] - center.unsqueeze(0)) / scale
            wpts[:, 1:, :3] = wpts[:, 1:, :3] / scale
        else :
            print(f"dataset type undefined : {self.type}")
            exit(-1)
            
        contact_point = points[0, :3].repeat(self.gt_trajs, 1)
        first_wpt = None
        if self.wpt_dim == 9 or (self.type == "absolute" and self.wpt_dim == 6):
            first_wpt = torch.hstack((contact_point, wpts[:, 0, 3:])) # use the second rot as the first rot
        elif self.wpt_dim == 3:
            first_wpt = contact_point

        if (self.wpt_dim == 9 or self.wpt_dim == 3 or (self.type == "absolute" and self.wpt_dim == 6)):
            if torch.sum(torch.abs(first_wpt[:,:3] - wpts[:, 0, :3])) / self.gt_trajs > 0.2: # magic number
                wpts = torch.cat([first_wpt.unsqueeze(1), wpts], dim=1)
            else :
                wpts[:, 0] = first_wpt
            wpts = wpts[:,:self.traj_len]
        
        # ret value
        return points, fusion, category, temp_wpts, wpts

if __name__=="__main__":
    
    dataset_dir = "../../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000/train"
    dataset = KptrajDeformAffordanceDataset(dataset_dir)
    dataset.print_data_shape()

