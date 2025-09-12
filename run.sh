


CUDA_VISIBLE_DEVICES=7 python test.py --network=./out/human_syn3/00010-/network-snapshot_002620.pkl  \
--pose_dist=./data/gen_human_2.npy  --output_path='./result/gen_samples/human3' \
--res=256 --truncation=0.7 --number=100 --type=gen_samples
#--is_normal True



