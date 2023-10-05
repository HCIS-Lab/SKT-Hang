import argparse, sys, json, yaml, cv2, imageio, os, time, glob
import open3d as o3d
import numpy as np

from datetime import datetime

from tqdm import tqdm
from time import strftime

from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pointnet2_ops.pointnet2_utils import furthest_point_sample
from utils.training_utils import get_model_module, get_dataset_module, optimizer_to_device, normalize_pc, kl_annealing

import torch
from torch.utils.data import DataLoader, Subset
from torchsummary import summary

from scipy.spatial.transform import Rotation as R
from utils.bullet_utils import draw_coordinate, get_matrix_from_pose, get_pos_rot_from_matrix, \
                                get_projmat_and_intrinsic, get_viewmat_and_extrinsic

from utils.testing_utils import refine_waypoint_rotation, robot_kptraj_hanging, recover_trajectory

def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    return train_set, val_set

def train(args):

    time_stamp = datetime.today().strftime('%m.%d.%H.%M')
    training_tag = time_stamp if args.training_tag == '' else f'{args.training_tag}'
    dataset_dir = args.dataset_dir
    dataset_root = args.dataset_dir.split('/')[-2] # dataset category
    dataset_subroot = args.dataset_dir.split('/')[-1] # time stamp
    config_file = args.config
    verbose = args.verbose
    device = args.device
    dataset_mode = 0 if 'absolute' in dataset_dir else 1 # 0: absolute, 1: residual

    config_file_id = config_file.split('/')[-1][:-5] # remove '.yaml'
    checkpoint_dir = f'{args.checkpoint_dir}/{config_file_id}-{training_tag}/{dataset_root}-{dataset_subroot}'
    print(f'checkpoint_dir: {checkpoint_dir}')
    os.makedirs(checkpoint_dir, exist_ok=True)

    config = None
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader) # dictionary

    # params for training
    dataset_name = config['dataset_module']
    dataset_class_name = config['dataset_class']
    module_name = config['module']
    model_name = config['model']
    model_inputs = config['model_inputs']
    dataset_inputs = config['dataset_inputs']
    train_traj_start = config['model_inputs']['train_traj_start']
    
    # training batch and iters
    batch_size = config['batch_size']
    start_epoch = config['start_epoch']
    stop_epoch = config['stop_epoch']

    # training scheduling params
    lr = config['lr']
    lr_decay_rate = config['lr_decay_rate']
    lr_decay_epoch = config['lr_decay_epoch']
    weight_decay = config['weight_decay']
    save_freq = config['save_freq']

    dataset_class = get_dataset_module(dataset_name, dataset_class_name)
    train_set = dataset_class(dataset_dir=f'{dataset_dir}/train', **dataset_inputs, device=args.device)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    sample_num_points = train_set.sample_num_points
    print(f'dataset: {dataset_dir}')
    print(f'checkpoint_dir: {checkpoint_dir}')
    print(f'num of points in point cloud: {sample_num_points}')

    network_class = get_model_module(module_name, model_name)
    network = network_class(**model_inputs, dataset_type=dataset_mode).to(device)
    
    if verbose:
        summary(network)

    # create optimizers
    network_opt = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)
    # learning rate scheduler
    network_lr_scheduler = torch.optim.lr_scheduler.StepLR(network_opt, step_size=lr_decay_epoch, gamma=lr_decay_rate)
    optimizer_to_device(network_opt, device)

    if verbose:
        print(f'training batches: {len(train_loader)}')

    # start training
    start_time = time.time()

    # will only work if 'kl_annealing' = 1 in "model_inputs"
    kl_weight = kl_annealing(kl_anneal_cyclical=True, 
                niter=stop_epoch, 
                start=0.0, stop=0.1, kl_anneal_cycle=5, kl_anneal_ratio=0.5)

    # train for every epoch
    for epoch in range(start_epoch, stop_epoch + 1):

        train_batches = enumerate(train_loader, 0)

        # training
        train_afford_losses = []
        train_dir_losses = []
        train_kl_losses = []
        train_recon_losses = []
        train_total_losses = []
        for i_batch, (sample_pcds, sample_affords, sample_trajs) in tqdm(train_batches, total=len(train_loader)):

            # set models to training mode
            network.train()

            # get segmented point cloud
            sample_pcds = sample_pcds.to(device).contiguous() 
            sample_trajs = sample_trajs.to(device).contiguous()
            sample_cp = sample_pcds[:, 0]

            # forward pass
            losses = network.get_loss(epoch, sample_pcds, sample_trajs, sample_cp, sample_affords, lbd_kl=kl_weight.get_beta())  # B x 2, B x F x N
            total_loss = losses['total']

            if 'afford' in losses.keys():
                train_afford_losses.append(losses['afford'].item())
            if 'dir' in losses.keys():
                train_dir_losses.append(losses['dir'].item())
            train_kl_losses.append(losses['kl'].item())
            train_recon_losses.append(losses['recon'].item())
            train_total_losses.append(losses['total'].item())

            # optimize one step
            network_opt.zero_grad()
            total_loss.backward()
            network_opt.step()

        network_lr_scheduler.step()
        
        if 'afford' in losses.keys():
            train_afford_avg_loss = np.mean(np.asarray(train_afford_losses))
        if 'dir' in losses.keys():
            train_dir_avg_loss = np.mean(np.asarray(train_dir_losses))
        train_kl_avg_loss = np.mean(np.asarray(train_kl_losses))
        train_recon_avg_loss = np.mean(np.asarray(train_recon_losses))
        train_total_avg_loss = np.mean(np.asarray(train_total_losses))
        print(
                f'''---------------------------------------------\n'''
                f'''[ training stage ]\n'''
                f''' - time : {strftime("%H:%M:%S", time.gmtime(time.time() - start_time)):>9s} \n'''
                f''' - epoch : {epoch:>5.0f}/{stop_epoch:<5.0f} \n'''
                f''' - lr : {network_opt.param_groups[0]['lr']:>5.2E} \n'''
                f''' - train_afford_avg_loss : {train_afford_avg_loss if 'afford' in losses.keys() else 0.0:>10.5f}\n'''
                f''' - train_dir_avg_loss : {train_dir_avg_loss if 'dir' in losses.keys() else 0.0:>10.5f}\n'''
                f''' - train_kl_avg_loss : {train_kl_avg_loss:>10.5f}\n'''
                f''' - train_recon_avg_loss : {train_recon_avg_loss:>10.5f}\n'''
                f''' - train_total_avg_loss : {train_total_avg_loss:>10.5f}\n'''
                f'''---------------------------------------------\n'''
            )
        
        # save checkpoint
        if (epoch - start_epoch) % save_freq == 0 and (epoch - start_epoch) > 0 and epoch > train_traj_start:
            with torch.no_grad():
                print('Saving checkpoint ...... ')
                # torch.save(network, os.path.join(checkpoint_dir, f'{sample_num_points}_points-network_epoch-{epoch}.pth'))
                torch.save(network.state_dict(), os.path.join(checkpoint_dir, f'{sample_num_points}_points-network_epoch-{epoch}.pth'))
                torch.save(network_opt.state_dict(), os.path.join(checkpoint_dir, f'{sample_num_points}_points-optimizer_epoch-{epoch}.pth'))
                torch.save(network_lr_scheduler.state_dict(), os.path.join(checkpoint_dir, f'{sample_num_points}_points-scheduler_epoch-{epoch}.pth'))

        kl_weight.update()

def capture_from_viewer(geometries):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    for geometry in geometries:
        vis.add_geometry(geometry)

    # Updates
    for geometry in geometries:
        vis.update_geometry(geometry)
    vis.poll_events()
    vis.update_renderer()

    o3d_screenshot_mat = vis.capture_screen_float_buffer(do_render=True) # need to be true to capture the image
    o3d_screenshot_mat = (255.0 * np.asarray(o3d_screenshot_mat)).astype(np.uint8)
    o3d_screenshot_mat = cv2.cvtColor(o3d_screenshot_mat,cv2.COLOR_BGR2RGB)
    o3d_screenshot_mat = cv2.resize(o3d_screenshot_mat, (o3d_screenshot_mat.shape[1] // 6, o3d_screenshot_mat.shape[0] // 6))
    vis.destroy_window()

    return o3d_screenshot_mat

def inference(args):

    import pybullet as p
    import pybullet_data
    from pybullet_robot_envs.envs.panda_envs.panda_env import pandaEnv

    # ================== config ==================

    checkpoint_dir = f'{args.checkpoint_dir}'
    config_file = args.config
    verbose = args.verbose
    visualize = args.visualize
    evaluate = args.evaluate
    device = args.device
    dataset_mode = 0 if 'absolute' in checkpoint_dir else 1 # 0: absolute, 1: residual
    weight_subpath = args.weight_subpath
    weight_path = f'{checkpoint_dir}/{weight_subpath}'

    assert os.path.exists(weight_path), f'weight file : {weight_path} not exists'

    checkpoint_subdir = checkpoint_dir.split('/')[1]
    checkpoint_subsubdir = checkpoint_dir.split('/')[2]

    config = None
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader) # dictionary


    assert os.path.exists(weight_path), f'weight file : {weight_path} not exists'
    print('=================================================')
    print(f'checkpoint: {weight_path}')
    print('=================================================')

    config = None
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader) # dictionary

    # params for network
    module_name = config['module']
    model_name = config['model']
    model_inputs = config['model_inputs']
    batch_size = config['batch_size']
    wpt_dim = config['dataset_inputs']['wpt_dim']
    sample_num_points = config['dataset_inputs']['sample_num_points']
    print(f'waypoint dimension = {wpt_dim}')
    print(f'num of points = {sample_num_points}')

    # inference
    inference_obj_dir = args.obj_shape_root
    assert os.path.exists(inference_obj_dir), f'{inference_obj_dir} not exists'
    inference_obj_whole_dirs = glob.glob(f'{inference_obj_dir}/*')

    inference_hook_shape_root = args.hook_shape_root
    assert os.path.exists(inference_hook_shape_root), f'{inference_hook_shape_root} not exists'

    inference_hook_dir = args.inference_dir # for hook shapes
    inference_hook_whole_dirs = glob.glob(f'{inference_hook_dir}/*')
    inference_hook_whole_dirs.sort()

    inference_obj_paths = []
    inference_hook_paths = []

    for inference_obj_path in inference_obj_whole_dirs:
        paths = glob.glob(f'{inference_obj_path}/*.json')
        assert len(paths) == 1, f'multiple object contact informations : {paths}'
        inference_obj_paths.extend(paths) 

    for inference_hook_path in inference_hook_whole_dirs:
        # if 'Hook' in inference_hook_path:
        paths = glob.glob(f'{inference_hook_path}/affordance-0.npy')
        inference_hook_paths.extend(paths) 

    obj_contact_poses = []
    obj_grasping_infos = []
    obj_urdfs = []
    obj_names = []
    for inference_obj_path in inference_obj_paths:
        obj_contact_info = json.load(open(inference_obj_path, 'r'))
        obj_contact_poses.append(obj_contact_info['contact_pose'])
        obj_grasping_infos.append(obj_contact_info['initial_pose'][0]) # bottom position

        obj_urdf = '{}/base.urdf'.format(os.path.split(inference_obj_path)[0])
        assert os.path.exists(obj_urdf), f'{obj_urdf} not exists'
        obj_urdfs.append(obj_urdf)

        obj_name = obj_urdf.split('/')[-2]
        obj_names.append(obj_name)

    hook_pcds = []
    hook_affords = []
    hook_urdfs = []

    class_num = 15 if ('/val' in inference_hook_dir) or '/test' in inference_hook_dir else 10000
    easy_cnt = 0
    normal_cnt = 0
    hard_cnt = 0
    devil_cnt = 0
    cnt = 0
    for inference_hook_path in inference_hook_paths:

        hook_name = inference_hook_path.split('/')[-2]
        points = np.load(inference_hook_path)[:, :3].astype(np.float32)
        affords = np.load(inference_hook_path)[:, 3].astype(np.float32)
        
        easy_cnt += 1 if 'easy' in hook_name else 0
        normal_cnt += 1 if 'normal' in hook_name else 0
        hard_cnt += 1 if 'hard' in hook_name else 0
        devil_cnt += 1 if 'devil' in hook_name else 0

        if 'easy' in hook_name and easy_cnt > class_num:
            continue
        if 'normal' in hook_name and normal_cnt > class_num:
            continue
        if 'hard' in hook_name and hard_cnt > class_num:
            continue
        if 'devil' in hook_name and devil_cnt > class_num:
            continue
    
        cnt += 1
        
        hook_urdf = f'{inference_hook_shape_root}/{hook_name}/base.urdf'
        assert os.path.exists(hook_urdf), f'{hook_urdf} not exists'
        hook_urdfs.append(hook_urdf) 
        hook_pcds.append(points)

    inference_subdir = os.path.split(inference_hook_dir)[-1]
    output_dir = f'inference/inference_trajs/{checkpoint_subdir}/{checkpoint_subsubdir}/{inference_subdir}'
    os.makedirs(output_dir, exist_ok=True)
    
    # ================== Model ==================

    # load model
    network_class = get_model_module(module_name, model_name)
    network = network_class(**model_inputs, dataset_type=dataset_mode).to(device)
    network.load_state_dict(torch.load(weight_path))

    # ================== Simulator ==================

    # Create pybullet GUI
    physics_client_id = None
    if visualize:
        physics_client_id = p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
    else:
        physics_client_id = p.connect(p.DIRECT)
    p.resetDebugVisualizerCamera(
        cameraDistance=0.2,
        cameraYaw=90,
        cameraPitch=-30,
        cameraTargetPosition=[0.5, 0.0, 1.3]
    )
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    sim_timestep = 1.0 / 240
    p.setTimeStep(sim_timestep)
    p.setGravity(0, 0, 0)

    cam_info = p.getDebugVisualizerCamera()
    width, height, view_mat, proj_mat = cam_info[0], cam_info[1], cam_info[2], cam_info[3]
    
    # ------------------- #
    # --- Setup robot --- #
    # ------------------- #

    # Load plane contained in pybullet_data
    p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"))
    robot = pandaEnv(physics_client_id, use_IK=1)

    # -------------------------- #
    # --- Load other objects --- #
    # -------------------------- #

    p.loadURDF(os.path.join(pybullet_data.getDataPath(), "table/table.urdf"), [1, 0.0, 0.0])

    # wall
    wall_pos = [0.5, -0.105, 0.93]
    wall_orientation = p.getQuaternionFromEuler([0, 0, 0])
    wall_id = p.loadURDF("../shapes/wall/wall.urdf", wall_pos, wall_orientation)

    # floor
    floor_pos = [0.0, 0.0, 0.0]
    floor_orientation = p.getQuaternionFromEuler([0, 0, 0])
    floor_id = p.loadURDF("../shapes/wall/floor.urdf", floor_pos, floor_orientation)

    hook_pose = [
        0.5,
        -0.1,
        1.3,
        4.329780281177466e-17,
        0.7071067811865475,
        0.7071067811865476,
        4.329780281177467e-17
    ]

    # ================== Inference ==================

    # batch_size = 10
    batch_size = 1
    all_scores = {
        'easy': [],
        'normal': [],
        'hard': [],
        'devil': [],
        'all': []
    }

    obj_sucrate = {}
    for k in obj_names:
        obj_sucrate[k] = {
            'easy': 0,
            'easy_all': 0,
            'normal': 0,
            'normal_all': 0,
            'hard': 0,
            'hard_all': 0,
            'devil': 0,
            'devil_all': 0,
        }

    for sid, pcd in enumerate(tqdm(hook_pcds)):

        # urdf file
        hook_urdf = hook_urdfs[sid]
        hook_id = p.loadURDF(hook_urdf, hook_pose[:3], hook_pose[3:])
        # p.resetBasePositionAndOrientation(hook_id, hook_pose[:3], hook_pose[3:])

        # hook name
        hook_name = hook_urdf.split('/')[-2]
        difficulty = 'easy' if 'easy' in hook_name else \
                     'normal' if 'normal' in hook_name else \
                     'hard' if 'hard' in hook_name else  \
                     'devil'
        
        # sample trajectories
        centroid_pcd, centroid, scale = normalize_pc(pcd, copy_pts=True) # points will be in a unit sphere
        contact_point = centroid_pcd[0]

        # If you want to see noise point cloud
        # centroid_pcd = 1.0 * (np.random.rand(pcd.shape[0], pcd.shape[1]) - 0.5).astype(np.float32) # random noise

        points_batch = torch.from_numpy(centroid_pcd).unsqueeze(0).to(device=device).contiguous()
        input_pcid = None
        point_num = points_batch.shape[1]
        if point_num >= sample_num_points:
            input_pcid = furthest_point_sample(points_batch, sample_num_points).long().reshape(-1)  # BN
        else :
            mod_num = sample_num_points % point_num
            repeat_num = int(sample_num_points // point_num)
            input_pcid = furthest_point_sample(points_batch, mod_num).long().reshape(-1)  # BN
            input_pcid = torch.cat([torch.arange(0, point_num).int().repeat(repeat_num).to(device), input_pcid])
        points_batch = points_batch[0, input_pcid, :].squeeze()
        points_batch = points_batch.repeat(batch_size, 1, 1)

        # generate trajectory using predicted contact points
        affordance, recon_trajs = network.sample(points_batch)

        # ###############################################
        # # =========== for affordance head =========== #
        # ###############################################

        points = points_batch[0].cpu().detach().squeeze().numpy()
        affordance = affordance[0].cpu().detach().squeeze().numpy()
        affordance = (affordance - np.min(affordance)) / (np.max(affordance) - np.min(affordance))
        colors = cv2.applyColorMap((255 * affordance).astype(np.uint8), colormap=cv2.COLORMAP_JET).squeeze()

        contact_point_cond = np.where(affordance == np.max(affordance))[0]
        contact_point = points[contact_point_cond][0]

        contact_point_coor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        contact_point_coor.translate(contact_point.reshape((3, 1)))

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors / 255)

        if visualize:
            img_list = []
            frames = 9
            rotate_per_frame = np.pi * 2 / frames
            for _ in range(frames):
                r = point_cloud.get_rotation_matrix_from_xyz((0, rotate_per_frame, 0)) # (rx, ry, rz) = (right, up, inner)
                point_cloud.rotate(r, center=(0, 0, 0))
                contact_point_coor.rotate(r, center=(0, 0, 0))
                geometries = [point_cloud, contact_point_coor]

                img = capture_from_viewer(geometries)
                img_list.append(img)
            
            save_path = f"{output_dir}/{weight_subpath[:-4]}-affor-{sid}.gif"
            imageio.mimsave(save_path, img_list, fps=10)

        ##############################################################
        # =========== for trajectory reconstruction head =========== #
        ##############################################################

        hook_poses = torch.Tensor(hook_pose).repeat(batch_size, 1).to(device)
        scales = torch.Tensor([scale]).repeat(batch_size).to(device)
        centroids = torch.from_numpy(centroid).repeat(batch_size, 1).to(device)
        recovered_trajs = recover_trajectory(recon_trajs, hook_poses, centroids, scales, dataset_mode, wpt_dim)

        # conting inference score using object and object contact information
        ignore_wpt_num = int(np.ceil(len(recovered_trajs[0]) * 0.1))
        if evaluate:
            max_obj_success_cnt = 0
            wpt_ids = []
            for traj_id, recovered_traj in enumerate(recovered_trajs):

                obj_success_cnt = 0
                for i, (obj_urdf, obj_contact_pose, obj_grasping_info) in enumerate(zip(obj_urdfs, obj_contact_poses, obj_grasping_infos)):
                    reversed_recovered_traj = recovered_traj[ignore_wpt_num:][::-1]
                    reversed_recovered_traj = refine_waypoint_rotation(reversed_recovered_traj)

                    obj_name = obj_urdf.split('/')[-2]

                    obj_id = p.loadURDF(obj_urdf)
                    rgbs, success = robot_kptraj_hanging(robot, reversed_recovered_traj, obj_id, hook_id, obj_contact_pose, obj_grasping_info, visualize=visualize if i == 0 else False)
                    res = 'success' if success else 'failed'
                    obj_sucrate[obj_name][difficulty] += 1 if success else 0
                    obj_sucrate[obj_name][f'{difficulty}_all'] += 1
                    obj_success_cnt += 1 if success else 0
                    p.removeBody(obj_id)

                    if len(rgbs) > 0 and traj_id == 0: # only when visualize=True
                        rgbs[0].save(f"{output_dir}/{weight_subpath[:-4]}-{sid}-{hook_name}-{res}.gif", save_all=True, append_images=rgbs, duration=80, loop=0)

                max_obj_success_cnt = max(obj_success_cnt, max_obj_success_cnt)

            print('[{} / {}] success rate: {:00.03f}%'.format(sid, hook_name, max_obj_success_cnt / len(obj_contact_poses) * 100))
            sys.stdout.flush()
            all_scores[difficulty].append(max_obj_success_cnt / len(obj_contact_poses))
            all_scores['all'].append(max_obj_success_cnt / len(obj_contact_poses))
        
        if visualize:

            obj_urdf = obj_urdfs[1]
            obj_id = p.loadURDF(obj_urdf)
            obj_contact_pose = obj_contact_poses[1]

            width, height = 640, 480
            fx = fy = 605
            far = 1000.
            near = 0.01
            projection_matrix, intrinsic = get_projmat_and_intrinsic(width, height, fx, fy, far, near)
            pcd_view_matrix, pcd_extrinsic = get_viewmat_and_extrinsic(cameraEyePosition=[0.8, 0.0, 1.3], cameraTargetPosition=[0.5, 0.0, 1.3], cameraUpVector=[0., 0., 1.])

            wpt_ids = []
            gif_frames = []
            for i, recovered_traj in enumerate(recovered_trajs):
                colors = list(np.random.rand(3)) + [1]
                for wpt_i, wpt in enumerate(recovered_traj[::-1]):
                    wpt_id = p.createMultiBody(
                        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, 0.001), 
                        baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, 0.001, rgbaColor=colors), 
                        basePosition=wpt[:3]
                    )
                    wpt_ids.append(wpt_id)

            for i, recovered_traj in enumerate(recovered_trajs):
                colors = list(np.random.rand(3)) + [1]
                for wpt_i, wpt in enumerate(recovered_traj[ignore_wpt_num:][::-1]):
                    
                    obj_tran = get_matrix_from_pose(wpt) @ np.linalg.inv(get_matrix_from_pose(obj_contact_pose))
                    obj_pos, obj_rot = get_pos_rot_from_matrix(obj_tran)
                    p.resetBasePositionAndOrientation(obj_id, obj_pos, obj_rot)

                    if wpt_i % 2 == 0:
                        img = p.getCameraImage(width, height, viewMatrix=pcd_view_matrix, projectionMatrix=projection_matrix)
                        rgb = np.reshape(img[2], (height, width, 4))[:,:,:3]
                        gif_frames.append(rgb)

            save_path = f"{output_dir}/{weight_subpath[:-4]}-{sid}-{hook_name}-{max_obj_success_cnt}.gif"
            imageio.mimsave(save_path, gif_frames, fps=10)

            for wpt_id in wpt_ids:
                p.removeBody(wpt_id)
            p.removeBody(obj_id)

        p.removeBody(hook_id)
        p.removeAllUserDebugItems()


    if evaluate:
        
        print("===============================================================================================")  # don't modify this
        print("success rate of all objects")
        for obj_name in obj_sucrate.keys():
            for difficulty in ['easy', 'normal', 'hard', 'devil']:
                assert difficulty in obj_sucrate[obj_name].keys() and f'{difficulty}_all' in obj_sucrate[obj_name].keys()
                print('[{}] {}: {:00.03f}%'.format(obj_name, difficulty, obj_sucrate[obj_name][difficulty] / obj_sucrate[obj_name][f'{difficulty}_all'] * 100))
        print("===============================================================================================")  # don't modify this

        easy_mean = np.asarray(all_scores['easy'])
        normal_mean = np.asarray(all_scores['normal'])
        hard_mean = np.asarray(all_scores['hard'])
        devil_mean = np.asarray(all_scores['devil'])
        all_mean = np.asarray(all_scores['all'])
        print("===============================================================================================")  # don't modify this
        print('checkpoint: {}'.format(weight_path))
        print('inference_dir: {}'.format(args.inference_dir))
        print('[easy] success rate: {:00.03f}%'.format(np.mean(easy_mean) * 100))
        print('[normal] success rate: {:00.03f}%'.format(np.mean(normal_mean) * 100))
        print('[hard] success rate: {:00.03f}%'.format(np.mean(hard_mean) * 100))
        print('[devil] success rate: {:00.03f}%'.format(np.mean(devil_mean) * 100))
        print('[all] success rate: {:00.03f}%'.format(np.mean(all_mean) * 100))
        print("===============================================================================================")  # don't modify this

def main(args):
    dataset_dir = args.dataset_dir
    checkpoint_dir = args.checkpoint_dir
    config_file = args.config

    if dataset_dir != '':
        assert os.path.exists(dataset_dir), f'{dataset_dir} not exists'
    if checkpoint_dir != '':
        assert os.path.exists(checkpoint_dir), f'{checkpoint_dir} not exists'
    if config_file != '':
        assert os.path.exists(config_file), f'{config_file} not exists'

    if args.training_mode == "train":
        train(args)

    if args.training_mode == "inference":
        inference(args)



if __name__=="__main__":

    parser = argparse.ArgumentParser()
    # about dataset
    parser.add_argument('--dataset_dir', '-dd', type=str, default='')

    # training mode
    parser.add_argument('--training_mode', '-tm', type=str, default='train', help="training mode : [train, inference]")
    parser.add_argument('--training_tag', '-tt', type=str, default='', help="training_tag")
    
    # testing
    parser.add_argument('--weight_subpath', '-wp', type=str, default='5000_points-network_epoch-150.pth', help="subpath of saved weight")
    parser.add_argument('--checkpoint_dir', '-cd', type=str, default='checkpoints', help="'training_mode=inference' only")
    parser.add_argument('--visualize', '-v', action='store_true')
    parser.add_argument('--evaluate', '-e', action='store_true')
    parser.add_argument('--inference_dir', '-id', type=str, default='')
    parser.add_argument('--obj_shape_root', '-osr', type=str, default='../shapes/inference_objs')
    parser.add_argument('--hook_shape_root', '-hsr', type=str, default='../shapes/hook_all_new_0')
    
    # other info
    parser.add_argument('--device', '-dv', type=str, default="cuda")
    parser.add_argument('--config', '-cfg', type=str, default='../config/modified_vatmart/modified_vatmart_3dof_40wpts.yaml')
    parser.add_argument('--split_ratio', '-sr', type=float, default=0.2)
    parser.add_argument('--verbose', '-vb', action='store_true')
    args = parser.parse_args()


    main(args)