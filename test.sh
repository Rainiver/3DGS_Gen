

CUDA_VISIBLE_DEVICES=7 python test.py --network=/data/vdd/zhongyuhe/workshop/AG3D/out/human_syn3_st2/00003-/network-snapshot_004636.pkl  \
--pose_dist=./data/test_human1.npy  --output_path='./result/gen_samples/human' \
--res=256 --truncation=0.7 --number=3 --type=gen_samples
#--is_normal True

# 模型只训了256的，--res只能选256