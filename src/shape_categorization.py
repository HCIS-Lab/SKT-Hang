

import os, glob, json, argparse, imageio
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import torch, cv2

from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
# from pointnet2_ops.pointnet2_utils import furthest_point_sample
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

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

def main(args):
    input_dataset = args.input_dataset
    traj_num = args.traj_num
    traj_dim = args.traj_dim
    wpt_dim = args.wpt_dim
    num_cls = args.num_cls
    
    assert os.path.exists(input_dataset), f'{input_dataset} not exists'
    assert os.path.exists(f'{input_dataset}/train'), f'{input_dataset}/train not exists'

    input_dataset_train = f'{input_dataset}/train'

    traj_name = input_dataset.split('/')[-2]
    traj_id = input_dataset.split('/')[-1]
    outdir = f'visualization/trajectory_pca/{traj_name}-{traj_id}-{traj_num}-{traj_dim}-{wpt_dim}'
    os.makedirs(outdir, exist_ok=True)

    hook_shape_dirs = glob.glob(f'{input_dataset_train}/*')

    hook_difficulties = []
    hook_trajectories = []
    hook_trajectories_raw = []
    hook_pcds = []
    hook_names = []
    for hook_shape_dir in tqdm(hook_shape_dirs):

        hook_name = hook_shape_dir.split('/')[-1]

        difficulty = 'easy' if 'easy' in hook_name else \
                    'normal' if 'normal' in hook_name else \
                    'hard' if 'hard' in hook_name else \
                    'devil'

        hook_pcd_path = glob.glob(f'{hook_shape_dir}/*.npy')[0]

        hook_traj_paths = glob.glob(f'{hook_shape_dir}/*.json')
        hook_traj_paths.sort(key=lambda x : int(x.split('/')[-1].split('-')[-1].split('.')[0])) # sort by trajectory id : [parent_dir]/traj-8.json => 8
        hook_traj_paths = hook_traj_paths[:traj_num]

        for hook_traj_path in hook_traj_paths:
            
            hook_names.append(hook_name)
            
            traj = json.load(open(hook_traj_path, 'r'))['trajectory']
            
            if wpt_dim == 3:
                traj_3d = np.asarray(traj)[:traj_dim, :3]
                hook_trajectories_raw.append(np.copy(traj_3d))
                traj_3d[:, :3] -= traj_3d[0, :3]
                hook_trajectories.append(traj_3d)

            if wpt_dim == 6:
                traj6d = np.asarray(traj)[:traj_dim]
                hook_trajectories_raw.append(np.copy(traj6d))
                traj6d[:, :3] -= traj6d[0, :3]
                hook_trajectories.append(traj6d)

            if wpt_dim == 9:
                traj_9d = np.zeros((traj_dim, 9))
                traj = np.asarray(traj)[:traj_dim]
                traj_rot = R.from_rotvec(traj[:, 3:]).as_matrix().reshape(-1, 9)[:, :6]
                traj_9d[:, 3:] = traj_rot
                hook_trajectories_raw.append(np.copy(traj_9d))
                traj_9d[:, :3] -= traj[0, :3]
                hook_trajectories.append(traj_9d)
            
            hook_pcds.append(hook_pcd_path)
            hook_difficulties.append(difficulty)
    
    hook_names = np.asarray(hook_names)
    hook_pcds = np.asarray(hook_pcds)
    hook_trajectories = np.asarray(hook_trajectories)
    hook_trajectories_raw = np.asarray(hook_trajectories_raw)
    hook_trajectories_reshape = np.reshape(hook_trajectories, (hook_trajectories.shape[0], -1))

    X_embedded = PCA(n_components=2).fit_transform(hook_trajectories_reshape)
    X_embedded = np.hstack((X_embedded, np.zeros((X_embedded.shape[0], 1)))).astype(np.float32)

    clustering = DBSCAN(eps=0.05, min_samples=100).fit(X_embedded)
    cond = np.where(clustering.labels_ >= 0)[0]

    hook_cnt = {hook_name: 0 for hook_name in hook_names}
    for hook_name in hook_names[cond]:
        hook_cnt[hook_name] += 1

    for key in hook_cnt.keys():
        if hook_cnt[key] == 0:
            print(f'{key} not included')
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot()

    kmeans = KMeans(n_clusters=num_cls)
    kmeans.fit(X_embedded[cond])
    new_dy = kmeans.predict(X_embedded[cond])
    ax.scatter(X_embedded[cond][:, 0],   X_embedded[cond][:, 1], c=new_dy, s=1)
    # for i in range(num_cls):
    #     print(f'{i} num: {len(np.where(new_dy == i)[0])}')

    hook_cls_res = {hook_name: {cls_id:0 for cls_id in range(num_cls)} for hook_name in hook_cnt.keys()}

    kmeans_centers = kmeans.cluster_centers_

    num_nn = 100
    nn = NearestNeighbors(n_neighbors=num_nn, algorithm='ball_tree').fit(X_embedded[cond])
    distances, indices = nn.kneighbors(kmeans_centers)
    
    for i, y in enumerate(new_dy):
        hook_cls_res[hook_names[cond][i]][y] += 1

    out_path = f'{args.input_dataset}/labels_{args.num_cls}c.txt'
    fout = open(out_path, 'w')

    fout.write('==============================================================================================\n')
    hook_cls = {}
    for key in hook_cls_res.keys():
        res = -1
        cnt = 0
        for c in hook_cls_res[key].keys():
            res = c if hook_cls_res[key][c] > cnt else res
            cnt = hook_cls_res[key][c] if hook_cls_res[key][c] > cnt else cnt

        fout.write(f'{key} => {res}\n')
        hook_cls[key] = res
    fout.write('==============================================================================================\n')

    for cls_id, indice in enumerate(indices):
        hook_names_in_cls = []
        for ind in indice:
            hook_name = hook_names[cond][ind]
            if hook_cls[hook_name] != cls_id:
                continue

            # print(f'class:{cls_id} => {hook_name} ({hook_subsets[hook_name]})')
            hook_names_in_cls.append(hook_name)
        
        if len(hook_names_in_cls) > 0:
            hook_names_in_cls = np.asarray(hook_names_in_cls)
            values, counts = np.unique(hook_names_in_cls, return_counts=True)
            mode = np.argmax(counts)
            # fout.write('===============================================\n')
            fout.write(f'center of class [{cls_id}]: {values[mode]}\n')
            # fout.write('===============================================\n')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dataset', '-id', type=str, default='../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview')
    parser.add_argument('--traj_num', '-tn', type=int, default=10000)
    parser.add_argument('--traj_dim', '-td', type=int, default=40)
    parser.add_argument('--wpt_dim', '-wd', type=int, default=3)
    parser.add_argument('--num_cls', '-nc', type=int, default=5)

    args = parser.parse_args()
    main(args)