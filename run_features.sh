export CUDA_VISIBLE_DEVICES=5
python extract_feature.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint /disk1/yliugu/Grounded-Segment-Anything/checkpoint/sam/groundingdino_swint_ogc.pth \
  --sam_checkpoint /disk1/yliugu/Grounded-Segment-Anything/checkpoint/sam/sam_vit_h_4b8939.pth \
  --input_image /disk1/yliugu/Grounded-Segment-Anything/data/nerf_llff_data/flower/images_8/image000.png \
  --output_dir /disk1/yliugu/Grounded-Segment-Anything/data/nerf_llff_data/flower/features \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --text_prompt "flower" \
  --device "cuda"