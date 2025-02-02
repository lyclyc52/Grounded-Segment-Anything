export CUDA_VISIBLE_DEVICES=6
python extract_feature.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint /disk1/yliugu/Grounded-Segment-Anything/checkpoint/sam/groundingdino_swint_ogc.pth \
  --sam_checkpoint /disk1/yliugu/Grounded-Segment-Anything/checkpoint/sam/sam_vit_h_4b8939.pth \
  --input_json_root /disk1/yliugu/Grounded-Segment-Anything/data/3dfront_0089/val \
  --output_dir /disk1/yliugu/Grounded-Segment-Anything/data/3dfront_0089/val/features \
  --device "cuda"