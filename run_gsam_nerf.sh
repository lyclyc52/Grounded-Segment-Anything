export CUDA_VISIBLE_DEVICES=7
python grounded_sam_nerf_demo.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint /disk1/yliugu/Grounded-Segment-Anything/checkpoint/sam/groundingdino_swint_ogc.pth \
  --sam_checkpoint /disk1/yliugu/Grounded-Segment-Anything/checkpoint/sam/sam_vit_h_4b8939.pth \
  --input_image /disk1/yliugu/torch-ngp/workspace/3dfront_0089_t0_results/train_results/ngp_ep0000_0113_rgb.png \
  --input_feature /disk1/yliugu/torch-ngp/workspace/3dfront_0089_t0_results/train_results/ngp_ep0000_0113_feature.npz \
  --output_dir outputs/3dfront_0089/nerf_train \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --text_prompt "sofa" \
  --device "cuda"