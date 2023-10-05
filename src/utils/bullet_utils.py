import numpy as np
import pybullet as p
import quaternion
from scipy.spatial.transform import Rotation as R

def xyzw2wxyz(quat : np.ndarray) -> np.ndarray:
    assert len(quat) == 4, f'quaternion size must be 4, got {len(quat)}'
    return np.asarray([quat[3], quat[0], quat[1], quat[2]])

def wxyz2xyzw(quat : np.ndarray) -> np.ndarray:
    assert len(quat) == 4, f'quaternion size must be 4, got {len(quat)}'
    return np.asarray([quat[1], quat[2], quat[3], quat[0]])

def pose_6d_to_7d(pose : list or tuple or np.ndarray) -> np.ndarray:
    assert len(pose) == 6, f'pose must contain 6 elements, but got {len(pose)}'
    pos = np.asarray(pose[:3])
    rot = R.from_rotvec(pose[3:]).as_quat()
    pose = list(pos) + list(rot)

    return np.array(pose)

def pose_6d_to_7d(pose : list or tuple or np.ndarray) -> np.ndarray:
    if len(pose) == 7:
        return np.array(pose)
    pos = np.asarray(pose[:3])
    rot = R.from_rotvec(pose[3:]).as_quat()
    pose_ret = list(pos) + list(rot)

    return np.array(pose_ret)

def pose_7d_to_6d(pose : list or tuple or np.ndarray) -> np.ndarray:
    if len(pose) == 6:
        return np.array(pose)
    pos = np.asarray(pose[:3])
    rot = R.from_quat(pose[3:]).as_rotvec()
    pose_ret = list(pos) + list(rot)

    return np.array(pose_ret)

def get_matrix_from_pose(pose : list or tuple or np.ndarray) -> np.ndarray:
    assert len(pose) == 6 or len(pose) == 7 or len(pose) == 9, f'pose must contain 6 or 7 elements, but got {len(pose)}'
    pos_m = np.asarray(pose[:3])
    rot_m = np.identity(3)

    if len(pose) == 6:
        rot_m = R.from_rotvec(pose[3:]).as_matrix()
    elif len(pose) == 7:
        rot_m = R.from_quat(pose[3:]).as_matrix()
    elif len(pose) == 9:
        rot_xy = pose[3:].reshape(2, 3)
        rot_m = np.vstack((rot_xy, np.cross(rot_xy[0], rot_xy[1]))).T
            
    ret_m = np.identity(4)
    ret_m[:3, :3] = rot_m
    ret_m[:3, 3] = pos_m

    return ret_m

def rot_6d_to_3d(rot) -> np.ndarray:

    rot_xy = np.asarray(rot)

    assert rot_xy.shape == (6,), f'dimension of rot should be (6,), but got {rot_xy.shape}'

    rot_xy = rot_xy.reshape(2, 3)
    rot_mat = np.vstack((rot_xy, np.cross(rot_xy[0], rot_xy[1]))).T 

    return R.from_matrix(rot_mat).as_rotvec()

def get_pose_from_matrix(matrix : list or tuple or np.ndarray, 
                        pose_size : int = 7) -> np.ndarray:

    mat = np.array(matrix)
    assert mat.shape == (4, 4), f'pose must contain 4 x 4 elements, but got {mat.shape}'
    
    pos = matrix[:3, 3]
    rot = None

    if pose_size == 6:
        rot = R.from_matrix(matrix[:3, :3]).as_rotvec()
    elif pose_size == 7:
        rot = R.from_matrix(matrix[:3, :3]).as_quat()
    elif pose_size == 9:
        rot = (matrix[:3, :2].T).reshape(-1)
            
    pose = list(pos) + list(rot)

    return np.array(pose)

def get_matrix_from_pos_rot(pos : list or tuple or np.ndarray, 
                            rot : list or tuple or np.ndarray) -> np.ndarray:
    assert (len(pos) == 3 and len(rot) == 4) or (len(pos) == 3 and len(rot) == 3)
    pos_m = np.asarray(pos)
    if len(rot) == 3:
        rot_m = R.from_rotvec(rot).as_matrix()
        # rot_m = np.asarray(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(rot))).reshape((3, 3))
    elif len(rot) == 4: # x, y, z, w
        rot_m = R.from_quat(rot).as_matrix()
        # rot_m = np.asarray(p.getMatrixFromQuaternion(rot)).reshape((3, 3))
    ret_m = np.identity(4)
    ret_m[:3, :3] = rot_m
    ret_m[:3, 3] = pos_m
    return ret_m


def cross(a:np.ndarray,b:np.ndarray)->np.ndarray:
    return np.cross(a,b)

def get_pos_rot_from_matrix(pose : np.ndarray) -> np.ndarray:
    assert pose.shape == (4, 4)
    pos = pose[:3, 3]
    rot = R.from_matrix(pose[:3, :3]).as_quat()
    return pos, rot

def get_projmat_and_intrinsic(width, height, fx, fy, far, near):

  cx = width / 2
  cy = height / 2
  fov = 2 * np.arctan(height / (2 * fy)) * 180.0 / np.pi

  project_matrix = p.computeProjectionMatrixFOV(
                      fov=fov,
                      aspect=width/height,
                      nearVal=near,
                      farVal=far
                    )
  
  intrinsic = np.array([
                [ fx, 0.0,  cx],
                [0.0,  fy,  cy],
                [0.0, 0.0, 1.0],
              ])
  
  return project_matrix, intrinsic

def get_viewmat_and_extrinsic(cameraEyePosition, cameraTargetPosition, cameraUpVector):

    view_matrix = p.computeViewMatrix(
                    cameraEyePosition=cameraEyePosition,
                    cameraTargetPosition=cameraTargetPosition,
                    cameraUpVector=cameraUpVector
                  )

    # rotation vector extrinsic
    z = np.asarray(cameraTargetPosition) - np.asarray(cameraEyePosition)
    norm = np.linalg.norm(z, ord=2)
    assert norm > 0, f'cameraTargetPosition and cameraEyePosition is at same location'
    z /= norm
   
    y = -np.asarray(cameraUpVector)
    y -= (np.dot(z, y)) * z
    norm = np.linalg.norm(y, ord=2)
    assert norm > 0, f'cameraUpVector is parallel to z axis'
    y /= norm
    
    x = cross(y, z)

    # extrinsic
    extrinsic = np.identity(4)
    extrinsic[:3, 0] = x
    extrinsic[:3, 1] = y
    extrinsic[:3, 2] = z
    extrinsic[:3, 3] = np.asarray(cameraEyePosition)

    return view_matrix, extrinsic

def draw_coordinate(pose : np.ndarray or tuple or list, size : float = 0.1, color : np.ndarray=np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])):
    assert (type(pose) == np.ndarray and pose.shape == (4, 4)) or (len(pose) == 7) or (len(pose) == 6)

    if len(pose) == 7 or len(pose) == 6:
        pose = get_matrix_from_pose(pose)

    origin = pose[:3, 3]
    x = origin + pose[:3, 0] * size
    y = origin + pose[:3, 1] * size
    z = origin + pose[:3, 2] * size
    p.addUserDebugLine(origin, x, color[0], 2, 0)
    p.addUserDebugLine(origin, y, color[1], 2, 0)
    p.addUserDebugLine(origin, z, color[2], 2, 0)

def draw_bbox(start : list or tuple or np.ndarray,
              end : list or tuple or np.ndarray):
    
    assert len(start) == 3 and len(end) == 3, f'infeasible size of position, len(position) must be 3'

    points_bb = [
        [start[0], start[1], start[2]],
        [end[0], start[1], start[2]],
        [end[0], end[1], start[2]],
        [start[0], end[1], start[2]],
        [start[0], start[1], end[2]],
        [end[0], start[1], end[2]],
        [end[0], end[1], end[2]],
        [start[0], end[1], end[2]],
    ]

    for i in range(4):
        p.addUserDebugLine(points_bb[i], points_bb[(i + 1) % 4], [1, 0, 0])
        p.addUserDebugLine(points_bb[i + 4], points_bb[(i + 1) % 4 + 4], [1, 0, 0])
        p.addUserDebugLine(points_bb[i], points_bb[i + 4], [1, 0, 0])