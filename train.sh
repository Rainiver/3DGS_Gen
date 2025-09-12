



#python train.py --outdir ./out/human_syn --cfg deepfashion \
#--data /data/vde/zhongyuhe/workshop/data/human_syn/ --gpus=6 --batch 12 --gamma 5 \
#--lambda_eikonal 0.1 --is_sdf True --lambda_normal 0.05 --mbstd-group=2 \
#--resume=/data/vde/zhongyuhe/workshop/AG3D/out/human_syn/network-snapshot_000201.pkl \
#--face_gan True --image_resolution 128

#python train.py --outdir ./out/human2 --cfg deepfashion \
#--data /data/vdd/zhongyuhe/workshop/tools/Multiview-Avatar-main/dataset/ --gpus=6 --batch 12 --gamma 5 \
#--lambda_eikonal 0.1 --is_sdf True --is_sr_module True --lambda_normal 0 --mbstd-group=2 \
#--face_gan True



#python train.py --outdir ./out/human2 --cfg deepfashion \
#--data /data/vdd/zhongyuhe/workshop/tools/Multiview-Avatar-main/dataset/ --gpus=3 --batch 6 --gamma 5 \
#--lambda_eikonal 0.1 --is_sdf True --is_sr_module True --lambda_normal 0 --mbstd-group=2 \
#--face_gan True --resume=/data/vdd/zhongyuhe/workshop/AG3D/out/human2/00031-/network-snapshot_000705.pkl

CUDA_VISIBLE_DEVICES=5 python train.py --outdir ./out/human_256 --cfg deepfashion \
--data /data/vdb/zhongyuhe/data/data/human_syn/ --gpus=1 --batch 4 --gamma 5 \
--lambda_eikonal 0.1 --is_sdf True --lambda_normal 0 --mbstd-group=2 \
  --image_resolution 256 --snap 1


#--resume=/data/vdd/zhongyuhe/workshop/AG3D/out/human_syn3/00010-/network-snapshot_002620.pkl
#--resume=/data/vdd/zhongyuhe/workshop/AG3D/out/human_syn3/00000-/network-snapshot_000201.pkl \


#python train.py --outdir ./out/human_syn3_st2 --cfg deepfashion \
#--data /data/vdd/zhongyuhe/workshop/dataset/human_syn_2/ --gpus=6 --batch 12 --gamma 5 \
#--is_sr_module True --is_sdf True --normal_gan True --lambda_eikonal 0.1 --mbstd-group=2 \
#--lambda_normal 0.05 --mbstd-group=2 --resume=/data/vdd/zhongyuhe/workshop/AG3D/out/human_syn3/00007-/network-snapshot_001209.pkl


