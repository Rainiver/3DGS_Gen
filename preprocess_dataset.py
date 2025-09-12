import os
import torch
# import trimesh
import glob
import json
import shutil
# import argparse
import pickle
import numpy as np
from scipy.spatial.transform import Rotation
from training.deformers.smplx import SMPL


# parser = argparse.ArgumentParser(
#     description='Preprocessing for AvatarRex.'
# )
# parser.add_argument('--data-dir', type=str, help='Directory that contains raw ZJU-MoCap data.')
# # parser.add_argument('--out-dir', type=str, help='Directory where preprocessed data is saved.')
# parser.add_argument('--seqname', type=str, default='CoreView_313', help='Sequence to process.')


def get_basedata(label):
    # data_dir = '/data/vde/zhongyuhe/workshop/data/human_syn'

    # body_model = BodyModel(bm_path='/data/vde/zhongyuhe/workshop/3dgs-avatar/body_models/smpl/neutral/model.pkl', num_betas=10, batch_size=1).cuda()
    body_model = SMPL('/data/vde/zhongyuhe/workshop/AG3D/training/deformers/smplx/SMPLX', gender='neutral')
    faces = np.load('/data/vde/zhongyuhe/workshop/3dgs-avatar/body_models/misc/faces.npz')['faces']

    # smpl_out_dir = os.path.join(out_dir, 'models')
    # if not os.path.exists(smpl_out_dir):
    #     os.makedirs(smpl_out_dir)

    # smpl_file = os.path.join(smpl_dir, '{}.npy'.format(idx))
    # verts_file = os.path.join(verts_dir, '{}.npy'.format(idx))

    # We only process SMPL parameters in world coordinate
    # params_path = os.path.join(data_dir, 'dataset.json')
    # with open(params_path, 'r') as f:
    #     smpl_params = json.load(f)

    # for i in range(7):
    # root_orient = Rotation.from_rotvec(np.array(params['global_orient'][d_idx]).reshape([-1])).as_matrix()
    # key = f'images_padding/img_000000_{i+1}.png'
    smpl_param = label[25:]
    cam_param = label[:25]

    global_orient = torch.from_numpy(np.array(smpl_param[4:7], dtype=np.float32)).reshape(1, 3)

    body_pose = torch.from_numpy(np.array(smpl_param[7:76], dtype=np.float32)).reshape(1, 69)
    transl = torch.from_numpy(np.array(smpl_param[1:4], dtype=np.float32)).reshape([1, 3])
    betas = torch.from_numpy(np.array(smpl_param[76:], dtype=np.float32)).reshape(1, 10)
    scale = torch.from_numpy(np.array(smpl_param[0], dtype=np.float32)).reshape(1, 1)

    # new_root_orient = Rotation.from_matrix(root_orient).as_rotvec().reshape([1, 3]).astype(np.float32)
    # new_root_orient = np.array(params['global_orient'][d_idx]).reshape([1, 3]).astype(np.float32)
    # new_trans = trans.reshape([1, 3]).astype(np.float32)
    #
    # new_root_orient_torch = torch.from_numpy(new_root_orient).cuda()
    # new_trans_torch = torch.from_numpy(new_trans).cuda()

    # Get shape vertices
    body = body_model(betas=betas)
    minimal_shape = body['vertices'].detach().cpu().numpy()  # template mesh
    #
    # Get bone transforms
    body = body_model(global_orient=global_orient, body_pose=body_pose,
                      betas=betas, transl=transl, scale=scale)

    # vertices = body['vertices'].detach().cpu().numpy()[0]
    # new_trans = new_trans + (verts - vertices).mean(0, keepdims=True)
    # new_trans_torch = torch.from_numpy(new_trans).cuda()
    #
    # body = body_model(root_orient=new_root_orient_torch, pose_body=pose_body_torch, pose_hand=pose_hand_torch,
    #                   betas=betas_torch, trans=new_trans_torch)
    #
    # Visualize SMPL mesh
    vertices = body['vertices'].detach().cpu().numpy()
    # mesh = trimesh.Trimesh(vertices=vertices, faces=body_model.faces)
    # out_filename = os.path.join(data_dir, '{:06d}.ply'.format(idx))
    # mesh.export(out_filename)
    pose_hand = body_pose[:, 63:]

    bone_transforms = body.A.detach().cpu().numpy()  # 对应为A
    # print('bone_transforms.shape', bone_transforms.shape)  # 应为 (24, 4, 4)

    Jtr_posed = body.joints.detach().cpu().numpy()

    # out_filename = os.path.join(smpl_out_dir, '{:06d}.npz'.format(idx))
    # out_filename = os.path.join(data_dir, f'models/{i + 1}.npz')

    return {minimal_shape,
            betas,
            Jtr_posed,
            bone_transforms,
            transl,
            global_orient,
            body_pose,
            pose_hand}

    # with open(os.path.join(out_dir, 'cam_params.json'), 'w') as f:
    #     json.dump(all_cam_params, f)
