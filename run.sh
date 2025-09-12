
#CUDA_VISIBLE_DEVICES=6 python test.py --network=./model/deep_fashion.pkl  --pose_dist=./data/dp_pose_dist.npy --output_path './result/deepfashion' --res=512 --truncation=0.7 --number=5 --type=gen_samples --is_mesh=True
#python test.py --network=./model/deep_fashion.pkl  --pose_dist=./data/dp_pose_dist.npy  --output_path='./result/gen_novel_view' --res=512 --truncation=0.7 --number=5 --type=gen_novel_view
#python test.py --network=./model/deep_fashion.pkl   --pose_dist=./data/dp_pose_dist.npy  --output_path='./result/result_interp' --res=512 --truncation=0.7 --number=5 --type=gen_interp --is_mesh=True
#python test.py --network=./model/deep_fashion.pkl   --pose_dist=./data/dp_pose_dist.npy  --output_path='./result/result_anim' --res=512 --truncation=0.7 --number=5 --type=gen_anim  --motion_path=./data/animation/motion_seq/crawl_forward_poses.npz

#CUDA_VISIBLE_DEVICES=2 python test.py --network=./model/deep_fashion.pkl  \
#--pose_dist=./data/dp_pose_dist.npy  --output_path='./result/gen_novel_view' \
#--res=512 --truncation=0.7 --number=5000 --type=gen_novel_view --is_mesh=True

CUDA_VISIBLE_DEVICES=7 python test.py --network=/data/vdd/zhongyuhe/workshop/AG3D/out/human_syn3/00010-/network-snapshot_002620.pkl  \
--pose_dist=/data/vdd/zhongyuhe/workshop/AG3D/data/gen_human_2.npy  --output_path='./result/gen_samples/human3' \
--res=256 --truncation=0.7 --number=100 --type=gen_samples
#--is_normal True



