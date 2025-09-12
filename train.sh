



CUDA_VISIBLE_DEVICES=0 python train.py --outdir ./out/human_256 --cfg deepfashion \
--data ./data/human_syn/ --gpus=1 --batch 4 --gamma 5 \
--lambda_eikonal 0.1 --is_sdf True --lambda_normal 0 --mbstd-group=2 \
  --image_resolution 256 --snap 1




