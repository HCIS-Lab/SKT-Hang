
import time
import torch
import numpy as np
import quaternion
import pybullet as p
from PIL import Image
from scipy.spatial.transform import Rotation as R

from pybullet_robot_envs.envs.panda_envs.panda_env import pandaEnv
from utils.bullet_utils import get_pose_from_matrix, get_matrix_from_pose, draw_coordinate, rot_6d_to_3d

PENETRATION_THRESHOLD = 0.0003 # 0.00003528

def augment_next_waypoint(waypoint : list or np.ndarray,
                                direction_vec : list or np.ndarray,
                                length : float,
                                aug_num : int=10,
                                # in degree
                                noise_pos : float=0.2,
                                # in degree
                                noise_rot : float=1) -> np.ndarray:

    assert len(waypoint) == 7 and len(direction_vec) == 3, \
        f'length of waypoint should be 7 and direction_vec should be 3 but got {len(waypoint)} and {len(direction_vec)}'
    
    base_pos    = np.asarray(waypoint[:3]) # + np.asarray([0.0, 0.0, 0.02]) for testing
    base_rotvec = R.from_quat(waypoint[3:]).as_rotvec()

    deg_to_rad = np.pi / 180.0

    pos_low_limit  = np.full((3,), -noise_pos * deg_to_rad)
    pos_high_limit = np.full((3,),  noise_pos * deg_to_rad)
    rot_low_limit  = np.full((3,), -noise_rot * deg_to_rad)
    rot_high_limit = np.full((3,),  noise_rot * deg_to_rad)

    step_direction_vec = np.zeros((3, aug_num))
    random_rot = R.from_rotvec(np.random.uniform(pos_low_limit, pos_high_limit, (aug_num, 3))).as_matrix()
    for i in range(aug_num):
        step_direction_vec[:, i] = (random_rot[i] @ direction_vec.reshape(3, 1)).reshape(3,)
    step_direction_vec = step_direction_vec.T
    # step_direction_vec = direction_vec + np.random.uniform(pos_low_limit, pos_high_limit, (aug_num, 3))
    # step_direction_vec /= np.linalg.norm(step_direction_vec, axis=1, ord=2)

    step_pos = base_pos + length * step_direction_vec
    step_rotvec = base_rotvec + np.random.uniform(rot_low_limit, rot_high_limit, (aug_num, 3)) \
                    if (base_rotvec <  np.pi - noise_rot * deg_to_rad).all() and \
                       (base_rotvec > -np.pi + noise_rot * deg_to_rad).all() \
                    else np.full((aug_num, 3),base_rotvec)
    step_quat = R.from_rotvec(step_rotvec).as_quat()
    step_pose = np.hstack((step_pos, step_quat))

    return step_pose

def penetration_score(hook_id : int, obj_id : int):

    p.performCollisionDetection()

    contact_points = p.getContactPoints(bodyA=hook_id, bodyB=obj_id)
    # closest_points = p.getClosestPoints(bodyA=hook_id, bodyB=obj_id, distance=thresh)
    # within_thresh = 1 if len(closest_points) > 0 else 0

    penetration = 0.0
    for contact_point in contact_points:
        # contact distance, positive for separation, negative for penetration
        contact_distance = contact_point[8] 
        penetration = min(penetration, contact_distance) if contact_distance < 0 else 0.0
    
    # return penetration, within_thresh
    return -penetration

def refine_waypoint_rotation(wpts : np.ndarray or list):

    assert wpts is not None and len(wpts) > 1, f'the trajectory only contains one waypoint or is None'

    rot_format = None
    if len(wpts[0]) == 6:
        rot_format = 'rotvec'
    elif len(wpts[0]) == 7:
        rot_format = 'quat'
    else:
        print('wrong waypoint format')
        exit(-1)

    # test direction
    next_pos = wpts[1][:3]
    tmp_pos = wpts[0][:3]
    tmp_dir = np.asarray(next_pos) - np.asarray(tmp_pos) 
    tmp_rot = wpts[0][3:]
    if rot_format == 'rotvec':
        tmp_rotmat = R.from_rotvec(tmp_rot).as_matrix()
    else :
        tmp_rotmat = R.from_quat(tmp_rot).as_matrix()
    tmp_rot_dir = (tmp_rotmat @ np.asarray([[1], [0], [0]])).T

    # no need to refine
    if np.dot(tmp_rot_dir, tmp_dir) > 0: 
        return wpts
    
    refine_mat = R.from_rotvec([0, 0, np.pi]).as_matrix()

    refined_wpts = []
    for i in range(len(wpts) - 1):
        tmp_pos = wpts[i][:3]
        tmp_rot = wpts[i][3:]
        if rot_format == 'rotvec':
            tmp_refined_rot = R.from_matrix(R.from_rotvec(tmp_rot).as_matrix() @ refine_mat).as_rotvec()
        else: 
            tmp_refined_rot = R.from_matrix(R.from_quat(tmp_rot).as_matrix() @ refine_mat).as_quat()
        tmp_refined_pose = list(tmp_pos) + list(tmp_refined_rot)
        refined_wpts.append(tmp_refined_pose)
        
    return refined_wpts

def trajectory_scoring(src_traj : list or np.ndarray, hook_id : int, obj_id : int, hook_pose : list or np.ndarray, obj_contact_pose : list or np.ndarray, visualize=False):

    if type(src_traj) == list:
        src_traj = np.asarray(src_traj)
    if type(obj_contact_pose) == list:
        obj_contact_pose = np.asarray(obj_contact_pose)

    assert obj_contact_pose.shape == (4, 4) or obj_contact_pose.shape == (7,), \
             f'the shape of obj_contact_pose must be (4, 4) or (7,), but got {obj_contact_pose.shape}'
    
    if obj_contact_pose.shape == (7,):
        obj_contact_trans = get_matrix_from_pose(obj_contact_pose)
    else :
        obj_contact_trans = obj_contact_pose
    
    hook_trans = get_matrix_from_pose(list(hook_pose))

    score = PENETRATION_THRESHOLD
    penetration_cost = 0.0
    color = np.random.rand(1, 3)
    color = np.repeat(color, 3, axis=0)
    rgbs = []
    cam_info = p.getDebugVisualizerCamera()
    for i, waypoint in enumerate(src_traj):

        relative_trans = get_matrix_from_pose(waypoint)
        world_trans = hook_trans @ relative_trans
        obj_trans = world_trans @ np.linalg.inv(obj_contact_trans)
        obj_pose = get_pose_from_matrix(obj_trans, pose_size=7)
        p.resetBasePositionAndOrientation(obj_id, obj_pose[:3], obj_pose[3:])

        draw_coordinate(world_trans, size=0.002)

        penetration = penetration_score(hook_id=hook_id, obj_id=obj_id)
        penetration_cost += penetration

        if visualize:
            width = cam_info[0]
            height = cam_info[1]
            view_mat = cam_info[2]
            proj_mat = cam_info[3]
            img_info = p.getCameraImage(width, height, viewMatrix=view_mat, projectionMatrix=proj_mat)
            rgb = img_info[2]
            rgbs.append(Image.fromarray(rgb))

        # score -= waypoint_penetration
        # within_thresh_cnt += within_thresh

    penetration_cost /= src_traj.shape[0]
    score = score - penetration_cost
    
    # score /= within_thresh_cnt if within_thresh_cnt != 0 else 1.0
    # score += PENETRATION_THRESHOLD # hyper param, < 0 : not good
    # ratio = score / PENETRATION_THRESHOLD

    p.removeAllUserDebugItems()

    return score, rgbs

def xyzw2wxyz(quat : np.ndarray) -> np.ndarray:
    assert len(quat) == 4, f'quaternion size must be 4, got {len(quat)}'
    return np.asarray([quat[3], quat[0], quat[1], quat[2]])

def wxyz2xyzw(quat : np.ndarray) -> np.ndarray:
    assert len(quat) == 4, f'quaternion size must be 4, got {len(quat)}'
    return np.asarray([quat[1], quat[2], quat[3], quat[0]])

def get_dense_waypoints(start_config : list or tuple or np.ndarray, end_config : list or tuple or np.ndarray, resolution : float=0.005):

    assert len(start_config) == 7 and len(end_config) == 7

    d12 = np.asarray(end_config) - np.asarray(start_config)
    steps = int(np.ceil(np.linalg.norm(np.divide(d12, resolution), ord=2)))
    obj_init_quat = quaternion.as_quat_array(xyzw2wxyz(start_config[3:]))
    obj_tgt_quat = quaternion.as_quat_array(xyzw2wxyz(end_config[3:]))

    ret = []
    # plan trajectory in the same way in collision detection module
    for step in range(steps):
        ratio = (step + 1) / steps
        pos = ratio * d12[:3] + np.asarray(start_config[:3])
        quat = quaternion.slerp_evaluate(obj_init_quat, obj_tgt_quat, ratio)
        quat = wxyz2xyzw(quaternion.as_float_array(quat))
        position7d = tuple(pos) + tuple(quat)
        ret.append(position7d)

    return ret

def robot_apply_action(robot : pandaEnv, obj_id : int, action : tuple or list, gripper_action : str = 'nop', 
                        sim_timestep : float = 1.0 / 240.0, diff_thresh : float = 0.005, max_vel : float = 0.2, max_iter = 5000):

    assert gripper_action in ['nop', 'pre_grasp', 'grasp']

    if gripper_action == 'nop':
        assert len(action) == 7, 'action length should be 7'

        robot.apply_action(action, max_vel=max_vel)
        diff = 10.0
        iter = 0
        while diff > diff_thresh and iter < max_iter:       
            iter += 1

            p.stepSimulation()
            time.sleep(sim_timestep)

            tmp_pos = p.getLinkState(robot.robot_id, robot.end_eff_idx, physicsClientId=robot._physics_client_id)[4] # position
            tmp_rot = p.getLinkState(robot.robot_id, robot.end_eff_idx, physicsClientId=robot._physics_client_id)[5] # rotation
            diff = np.sum((np.array(tmp_pos + tmp_rot) - np.array(action)) ** 2) ** 0.5

    elif gripper_action == 'pre_grasp' :

        robot.pre_grasp()
        for _ in range(int(1.0 / sim_timestep) * 1): # 1 sec
            p.stepSimulation()
            time.sleep(sim_timestep)
    else:

        robot.grasp(obj_id)
        for _ in range(int(1.0 / sim_timestep)): # 1 sec
            p.stepSimulation()
            time.sleep(sim_timestep)

def refine_rotation(src_transform, tgt_transform):
    src_rot = src_transform[:3, :3]
    tgt_rot = tgt_transform[:3, :3]

    src_rotvec = R.from_matrix(src_rot).as_rotvec()
    tgt_rotvec = R.from_matrix(tgt_rot).as_rotvec()

    rot_180 = np.identity(4)
    rot_180[:3, :3] = R.from_rotvec([0, 0, np.pi]).as_matrix()
    tgt_dual_transform = tgt_transform @ rot_180
    tgt_dual_rotvec = R.from_matrix(tgt_dual_transform[:3, :3]).as_rotvec()

    return tgt_transform if np.sum((src_rotvec - tgt_rotvec) ** 2) < np.sum((src_rotvec - tgt_dual_rotvec) ** 2) else tgt_dual_transform

def recover_trajectory(traj_src : torch.Tensor or np.ndarray, hook_poses : torch.Tensor or np.ndarray, 
                        centers : torch.Tensor or np.ndarray, scales : torch.Tensor or np.ndarray, dataset_mode : int=0, wpt_dim : int=6):
    # traj : dim = batch x num_steps x 6
    # dataset_mode : 0 for abosute, 1 for residual 

    traj = None
    if type(traj_src) == torch.Tensor:
        traj = traj_src.clone().cpu().detach().numpy()
        centers = centers.clone().cpu().detach().numpy()
        scales = scales.clone().cpu().detach().numpy()
        hook_poses = hook_poses.clone().cpu().detach().numpy()
    elif type(traj_src) == np.ndarray:
        traj = np.copy(traj_src)


    waypoints = []

    if dataset_mode == 0: # "absolute"

        for traj_id in range(traj.shape[0]):
            traj[traj_id, :, :3] = traj[traj_id, :, :3] * scales[traj_id] + centers[traj_id]
        
        for traj_id in range(traj.shape[0]): # batches
            waypoints.append([])
            hook_trans = get_matrix_from_pose(hook_poses[traj_id])
            for wpt_id in range(0, traj[traj_id].shape[0]): # waypoints

                wpt = np.zeros(6)
                if wpt_dim == 6 or wpt_dim == 9:

                    wpt = traj[traj_id, wpt_id]
                    current_trans = hook_trans @ get_matrix_from_pose(wpt)

                elif wpt_dim == 3:
                    # contact pose rotation
                    wpt[:3] = traj[traj_id, wpt_id]

                    # transform to world coordinate first
                    current_trans = np.identity(4)
                    current_trans[:3, 3] = traj[traj_id, wpt_id]
                    current_trans = hook_trans @ current_trans

                    if wpt_id < traj[traj_id].shape[0] - 1:
                        # transform to world coordinate first

                        peep_num_max = int(np.ceil(traj[traj_id].shape[0] / 4.0))
                        peep_num = peep_num_max if wpt_id < traj[traj_id].shape[0] - peep_num_max else traj[traj_id].shape[0] - wpt_id - 1
                        to_pos = np.ones((4, peep_num))
                        to_pos[:3] = traj[traj_id, wpt_id:wpt_id+peep_num].T 
                        to_pos = (hook_trans @ to_pos)[:3]
                        
                        from_pos = np.ones((4, peep_num))
                        from_pos[:3] = traj[traj_id, wpt_id+1:wpt_id+peep_num+1].T 
                        from_pos = (hook_trans @ from_pos)[:3]

                        weight = np.array([1/x for x in range(3, peep_num+3)])[:peep_num]
                        weight /= np.sum(weight)
                        diff = (to_pos - from_pos) * weight
                        
                        x_direction = np.sum(diff, axis=1)
                        x_direction /= np.linalg.norm(x_direction, ord=2)
                        y_direction = np.cross(x_direction, [0, 0, -1])
                        y_direction /= np.linalg.norm(y_direction, ord=2)
                        z_direction = np.cross(x_direction, y_direction)
                        rotation_mat = np.vstack((x_direction, y_direction, z_direction)).T
                        current_trans[:3, :3] = rotation_mat
                        
                    else :

                        current_trans[:3, :3] = R.from_rotvec(waypoints[-1][-1][3:]).as_matrix() # use the last waypoint's rotation as current rotation
                
                waypoints[-1].append(get_pose_from_matrix(current_trans, pose_size=6))
    
    if dataset_mode == 1: # "residual"

        for traj_id in range(traj.shape[0]):
            traj[traj_id,  0, :3] = (traj[traj_id,  0, :3] * scales[traj_id]) + centers[traj_id]
            traj[traj_id, 1:, :3] = (traj[traj_id, 1:, :3] * scales[traj_id])

        for traj_id in range(traj.shape[0]):
            waypoints.append([])
            tmp_pos = np.array([0.0, 0.0, 0.0])
            tmp_rot = np.array([0.0, 0.0, 0.0])
            hook_trans = get_matrix_from_pose(hook_poses[traj_id])
            for wpt_id in range(0, traj[traj_id].shape[0]):
                
                wpt = np.zeros(6)
                if wpt_dim == 6 or wpt_dim == 9:


                    if wpt_id == 0 :
                        wpt_tmp = traj[traj_id, wpt_id]
                        tmp_pos = wpt_tmp[:3]
                        tmp_rot = wpt_tmp[3:] if wpt_dim == 6 else rot_6d_to_3d(wpt_tmp[3:])
                        wpt[:3] = tmp_pos
                        wpt[3:] = tmp_rot

                    else :
                        tmp_pos = tmp_pos + np.asarray(traj[traj_id, wpt_id, :3])
                        tmp_rot = R.from_matrix(
                                    R.from_rotvec(
                                        traj[traj_id, wpt_id, 3:] if wpt_dim == 6 else rot_6d_to_3d(traj[traj_id, wpt_id, 3:])
                                    ).as_matrix() @ R.from_rotvec(
                                        tmp_rot
                                    ).as_matrix()
                                ).as_rotvec()
                        wpt[:3] = tmp_pos
                        wpt[3:] = tmp_rot
                        
                    current_trans = hook_trans @ get_matrix_from_pose(wpt)
                
                elif wpt_dim == 3 :
                    
                    # transform to world coordinate first
                    current_trans = np.identity(4)
                    current_trans[:3, 3] = tmp_pos + traj[traj_id, wpt_id]
                    current_trans = hook_trans @ current_trans
                    tmp_pos += traj[traj_id, wpt_id]

                    if wpt_id < traj[traj_id].shape[0] - 1:
                        # transform to world coordinate first
                        peep_num_max = int(np.ceil(traj[traj_id].shape[0] / 4.0))
                        peep_num = peep_num_max if wpt_id < traj[traj_id].shape[0] - peep_num_max else traj[traj_id].shape[0] - wpt_id - 1
                        to_pos = np.ones((3, peep_num))
                        to_pos[:3] = traj[traj_id, wpt_id+1:wpt_id+peep_num+1].T
                        to_pos = hook_trans[:3, :3] @ to_pos

                        weight = np.array([1/x for x in range(3, peep_num+3)])[:peep_num]
                        weight /= np.sum(weight)
                        diff = to_pos * weight

                        x_direction = np.sum(diff, axis=1)
                        x_direction /= np.linalg.norm(-x_direction, ord=2)
                        y_direction = np.cross(x_direction, [0, 0, -1])
                        y_direction /= np.linalg.norm(y_direction, ord=2)
                        z_direction = np.cross(x_direction, y_direction)
                        rotation_mat = np.vstack((x_direction, y_direction, z_direction)).T
                        current_trans[:3, :3] = rotation_mat
                    else :
                        current_trans[:3, :3] = R.from_rotvec(waypoints[-1][-1][3:]).as_matrix() # use the last waypoint's rotation as current rotation

                waypoints[-1].append(get_pose_from_matrix(current_trans, pose_size=6))
    
    return waypoints

def robot_kptraj_hanging(robot : pandaEnv, recovered_traj, obj_id, hook_id, contact_pose, grasping_info, sim_timestep=1.0/240, visualize=False):

    height_thresh = 0.8
    obj_contact_relative_transform = get_matrix_from_pose(contact_pose)

    obj_pose = grasping_info['obj_pose']
    obj_transform = get_matrix_from_pose(obj_pose)
    
    robot_pose = grasping_info['robot_pose']
    robot_transform = get_matrix_from_pose(robot_pose)

    robot.reset()

    # rendering
    width=320
    height=240
    far = 1.
    near = 0.01
    fov = 90.
    aspect_ratio = 1.
    cameraEyePosition=[0.75, 0.1, 1.3]
    cameraTargetPosition=[0.5, 0.1, 1.3]
    cameraUpVector=[0.0, 0.0, 1.0]
    view_mat = p.computeViewMatrix(
        cameraEyePosition=cameraEyePosition,
        cameraTargetPosition=cameraTargetPosition,
        cameraUpVector=cameraUpVector,
    )
    proj_mat = p.computeProjectionMatrixFOV(
        fov, aspect_ratio, near, far
    )

    # grasp the objct
    robot.apply_action(robot_pose, max_vel=-1)
    for _ in range(int(1.0 / sim_timestep * 0.5)): 
        p.stepSimulation()
        time.sleep(sim_timestep)
    p.resetBasePositionAndOrientation(obj_id, obj_pose[:3], obj_pose[3:])
    robot.grasp(obj_id=obj_id)
    for _ in range(int(1.0 / sim_timestep * 0.25)): 
        p.resetBasePositionAndOrientation(obj_id, obj_pose[:3], obj_pose[3:])
        p.stepSimulation()
        time.sleep(sim_timestep)

    # first kpt pose
    first_kpt_transform_world = get_matrix_from_pose(recovered_traj[0])

    # first object pose
    first_obj_kpt_transform_world = obj_transform @ obj_contact_relative_transform
    first_obj_kpt_transform_world = refine_rotation(first_kpt_transform_world, first_obj_kpt_transform_world)

    # first robot pose
    kpt_to_gripper = np.linalg.inv(first_obj_kpt_transform_world) @ robot_transform
    first_gripper_pose = get_pose_from_matrix(first_kpt_transform_world @ kpt_to_gripper)

    # move to the first waypoint
    trajectory_start = get_dense_waypoints(robot_pose, first_gripper_pose, resolution=0.002 )
    for waypoint in trajectory_start:
        robot.apply_action(waypoint)
        p.stepSimulation()
        robot.grasp()
        for _ in range(3): 
            p.stepSimulation()
            time.sleep(sim_timestep)

    rgbs = []
    # cam_info = p.getDebugVisualizerCamera()
    # width, height, view_mat, proj_mat = cam_info[0], cam_info[1], cam_info[2], cam_info[3]

    # colors = list(np.random.rand(3)) + [1]
    # wpt_ids = []
    old_gripper_pose = first_gripper_pose
    for i, waypoint in enumerate(recovered_traj):

        gripper_transform = get_matrix_from_pose(waypoint) @ kpt_to_gripper
        gripper_pose = get_pose_from_matrix(gripper_transform)

        fine_gripper_poses = get_dense_waypoints(old_gripper_pose, gripper_pose, resolution=0.002)
        for fine_gripper_pose in fine_gripper_poses:
            robot.apply_action(fine_gripper_pose)
            p.stepSimulation()
            
            robot.grasp()
            for _ in range(3): 
                p.stepSimulation()
                time.sleep(sim_timestep)

            if visualize:
                # wpt_id = p.createMultiBody(
                #     baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, 0.002), 
                #     baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, 0.002, rgbaColor=colors), 
                #     basePosition=waypoint[:3]
                # )
                # wpt_ids.append(wpt_id)
                img_info = p.getCameraImage(width, height, viewMatrix=view_mat, projectionMatrix=proj_mat)
                rgb = img_info[2]
                rgbs.append(rgb)
                
        old_gripper_pose = gripper_pose

    
    # for wpt_id in wpt_ids:
    #     p.removeBody(wpt_id)

    # release gripper
    robot.pre_grasp()
    for i in range(100): # 1 sec
        p.stepSimulation()
        time.sleep(sim_timestep)

    if visualize:
        img_info = p.getCameraImage(width, height, viewMatrix=view_mat, projectionMatrix=proj_mat)
        rgb = img_info[2]
        rgbs.append(rgb)

    # go to the ending pose
    gripper_rot = p.getLinkState(robot.robot_id, robot.end_eff_idx, physicsClientId=robot._physics_client_id)[5]
    gripper_rot_matrix = R.from_quat(gripper_rot).as_matrix()
    ending_gripper_pos = np.asarray(gripper_pose[:3]) + (gripper_rot_matrix @ np.array([[0], [0], [-0.05]])).reshape(3)
    action = tuple(ending_gripper_pos) + tuple(gripper_rot)

    robot_apply_action(robot, obj_id, action, gripper_action='nop', 
        sim_timestep=0.05, diff_thresh=0.005, max_vel=-1, max_iter=100)

    if visualize:
        img_info = p.getCameraImage(width, height, viewMatrix=view_mat, projectionMatrix=proj_mat)
        rgb = img_info[2]
        rgbs.append(rgb)
    
    # left force
    p.setGravity(2, 0, -5)
    for _ in range(1000):
        pos, rot = p.getBasePositionAndOrientation(obj_id)
        if pos[2] < height_thresh:
            break
        p.stepSimulation()

    # right force
    p.setGravity(-2, 0, -5)
    for _ in range(1000):
        pos, rot = p.getBasePositionAndOrientation(obj_id)
        if pos[2] < height_thresh:
            break
        p.stepSimulation()

    rgbs = [Image.fromarray(rgb) for rgb in rgbs]
    success = True if p.getContactPoints(obj_id, hook_id) != () else False
    return rgbs, success