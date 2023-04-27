export CUDA_VISIBLE_DEVICES=6
python grounded_sam_demo.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint /disk1/yliugu/Grounded-Segment-Anything/checkpoint/sam/groundingdino_swint_ogc.pth \
  --sam_checkpoint /disk1/yliugu/Grounded-Segment-Anything/checkpoint/sam/sam_vit_h_4b8939.pth \
  --input_image /disk1/yliugu/Grounded-Segment-Anything/data/3dfront_0089/train/images/0113.jpg\
  --output_dir outputs/3dfront_0089/gsam_train \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --text_prompt "sofa" \
  --device "cuda"