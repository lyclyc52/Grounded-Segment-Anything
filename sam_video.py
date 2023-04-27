import torch 
import numpy as np 
import json
import os
import cv2
import imageio
import tqdm
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from sam_demo import show_points, show_mask


def main():
    intrinsics = np.array([400., 400., 320., 240.])
    sam_checkpoint = "/disk1/yliugu/Grounded-Segment-Anything/checkpoint/sam/sam_vit_h_4b8939.pth"
    device = "cuda:7"
    model_type = "default"
    threshold = 0.2
    frame_root = '/disk1/yliugu/torch-ngp/workspace/3dfront_0089_t0_results/results/frames'
    pose_file = '/disk1/yliugu/torch-ngp/workspace/3dfront_0089_t0_results/results/poses.json'
    output_root = '/disk1/yliugu/Grounded-Segment-Anything/outputs/3dfront_0089/video'
    input_point = np.array([[100, 470]])
    input_label = np.array([1])


    os.makedirs(output_root, exist_ok=True)
    with open(pose_file) as f:
        poses = json.load(f)['poses']
    
    images = []
    depths = []
    features = []
    for i in tqdm.tqdm(range(len(poses))):
        rgb_name = os.path.join(frame_root, f'ngp_ep0000_{i:04}_rgb.png')
        depth_name = os.path.join(frame_root, f'ngp_ep0000_{i:04}_depth.npz')
        feature_name = os.path.join(frame_root, f'ngp_ep0000_{i:04}_feature.npz')
        image = cv2.imread(rgb_name, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        
        with np.load(depth_name) as data:
            size = data['size']
            depth = data['depth']
            depth = depth.reshape(size.tolist()).astype(np.float32)
            depths.append(depth)
        
        with np.load(feature_name) as data:
            res = data['res']
            feature = data['embedding']
            feature = feature.reshape(res.tolist()).astype(np.float32)
            features.append(feature)
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    pts_3D = None
    predictor.set_image(images[0])
    for i, (p,r,d,f) in tqdm.tqdm(enumerate(zip(poses, images, depths, features))):
        if i == 0:
            pts_3D = project_to_3d(input_point, p, intrinsics, d)

        pts_2D, pts_depth = project_to_2d(pts_3D, p, intrinsics)
        im_depth = d[pts_2D[...,1], pts_2D[..., 0]]
        valid = (pts_depth > 0.) and (torch.abs(im_depth- pts_depth) < threshold )
        pts_2D = pts_2D[valid]
        predictor.set_torch_feature(f)
        
        masks, scores, logits = predictor.predict(
            point_coords=pts_2D.numpy(),
            point_labels=input_label,
            multimask_output=True,
        )

        
        for j, (mask, score) in enumerate(zip(masks, scores)):
            output_file = os.path.join(output_root, f'{i:04d}_masks_{j}.png')
            plt.figure(figsize=(10,10))
            plt.imshow(r)
            show_mask(mask, plt.gca())
            show_points(pts_2D, input_label, plt.gca())
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.savefig(output_file)
            plt.close()
        


def project_to_3d(pts, pose, intrinsics, depth):
    '''
    Args:
        pts: Nx2
        pose: 4x4
        intrinsics: fx, fy, cx, cy
        depth: HxW
    '''
    pts = torch.from_numpy(pts)
    pose = torch.tensor(pose)
    fx, fy, cx, cy = intrinsics
    zs = torch.ones_like(pts[..., 0])
    xs = (pts[..., 0] - cx) / fx * zs
    ys = (pts[..., 1]  - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    
    pts_z = depth[pts[..., 1], pts[..., 0]] 
    directions = directions * pts_z
    
    rays_d = directions @ pose[:3, :3].transpose(1,0) # (N, 3)
    rays_o = pose[:3, 3] # [3]
    rays_o = rays_o[None, :]
    return rays_o + rays_d

def project_to_2d(pts, pose, intrinsics):
    fx, fy, cx, cy = intrinsics
    pose = torch.tensor(pose)
    pose = torch.inverse(pose)
    
    camera_pts = pts @ pose[:3, :3].T
    camera_pts = camera_pts + pose[:3,3]

    camera_x = camera_pts[..., 0] / camera_pts[..., -1] * fx + cx
    camera_y = camera_pts[..., 1] / camera_pts[..., -1] * fy + cy
    
    pts_depth = torch.norm(camera_pts, dim=-1)
    sign = torch.ones_like(pts_depth)
    sign[camera_pts[..., -1] < 0.] = -1.
    
    return torch.stack([camera_x, camera_y], dim = -1).to(torch.int), pts_depth * sign

def create_video():
    input_path = '/disk1/yliugu/Grounded-Segment-Anything/outputs/3dfront_0089/video'
    save_path = '/disk1/yliugu/Grounded-Segment-Anything/outputs/3dfront_0089/video'
    all_preds = []
    index = 0
    for i in range(11):
        image = cv2.imread(os.path.join(input_path, f'{i:04d}_masks_{index}.png'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        all_preds.append(image)
    
    imageio.mimwrite(os.path.join(save_path, f'rgb_{index}.mp4'), all_preds, fps=25, quality=8, macro_block_size=1)
    

if __name__ == '__main__':
    # main()
    create_video()