import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
from segment_anything import sam_model_registry, SamPredictor

def main():
    # Control Flag
    sam_checkpoint = "/disk1/yliugu/Grounded-Segment-Anything/checkpoint/sam/sam_vit_h_4b8939.pth"
    device = "cuda:7"
    model_type = "default"
    image_path = '/disk1/yliugu/torch-ngp/workspace/3dfront_0089_t0_results/test_results/ngp_ep0000_0006_rgb.png'
    # image_path = '/disk1/yliugu/torch-ngp/workspace/3dfront_0089_t0_results/test_results/ngp_ep0000_0004_rgb.png'
    # image_path = '/disk1/yliugu/torch-ngp/workspace/3dfront_0089_t0_results/train_results/ngp_ep0000_0113_rgb.png'
    output_root = 'outputs/test'
    feature = None
    # feature = '/disk1/yliugu/torch-ngp/workspace/3dfront_0089_t0_results/results/ngp_ep0000_0004_feature.npz'
    feature = '/disk1/yliugu/torch-ngp/workspace/3dfront_0089_t0_results/test_results/ngp_ep0000_0006_feature.npz'

    mode = 'point'
    input_point = np.array([[400, 370]])
    input_label = np.array([1])
    
    
    
    os.makedirs(output_root, exist_ok=True)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    output_file = os.path.join(output_root, 'prompt_img.png')
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_points(input_point, input_label, plt.gca())
    plt.axis('on')
    plt.savefig(output_file) 


    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(image)


    if feature is not None:
        with np.load(feature) as data:
            res = data['res']
            feature = data['embedding']
            feature = feature.reshape(res.tolist())
        predictor.set_torch_feature(feature)

    if mode == 'point':
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

    for i, (mask, score) in enumerate(zip(masks, scores)):
        output_file = os.path.join(output_root, f'masks_{i}.png')
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.savefig(output_file)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)



if __name__=='__main__':
    main()