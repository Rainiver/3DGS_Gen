# import zipfile
import math
import numpy as np
import torch
# from utils.graphics_utils import getProjectionMatrixTensor, fov2focal, focal2fov
from scene.cameras import MiniCam
from scipy.spatial.transform import Rotation
from preprocess_dataset import get_basedata
from training.deformers.smplx import SMPL
import trimesh
from plyfile import PlyData, PlyElement
from typing import NamedTuple


# def parse_raw_labels_ffhq(label_arr: torch.Tensor):
#     """Parse raw labels from FHHQ dataset
#     see `notebooks/merge_ffhq_flame_to_cam_json.py` for details.
#     """
#     device = label_arr.device
#     extrinsic = label_arr[..., :16].reshape(*label_arr.shape[:-1], 4, 4)
#     intrinsics = label_arr[..., 16:25]
#     fov_deg = 18.837
#     FovX = FovY = np.deg2rad(fov_deg)
#     intrinsics = intrinsics.reshape(*label_arr.shape[:-1], 3, 3)
#     T = torch.inverse(extrinsic)[..., :3, 3]
#     betas = label_arr[..., 25:125]
#     global_orient = label_arr[..., 125:128]
#     jaw_pose = label_arr[..., 128:131]
#     expression = label_arr[..., 131:]
#     assert expression.shape[-1] == 50
#     return dict(
#         cam2world=extrinsic,
#         T=T,
#         fovx=torch.tensor(FovX, device=device, dtype=torch.float32).reshape(*label_arr.shape[:-1], 1),
#         fovy=torch.tensor(FovY, device=device, dtype=torch.float32).reshape(*label_arr.shape[:-1], 1),
#         principle_points=intrinsics[..., :2, 2],
#         betas=betas,
#         expression=expression,
#         global_orient=global_orient,
#         jaw_pose=jaw_pose,
#     )
#
# def update_smpl_to_raw_labels(label_arr, global_orient=None, body_pose=None, betas=None):
#     base_idx = 25
#     label_arr = label_arr.clone()
#     if global_orient is not None:
#         label_arr[..., base_idx:3+base_idx] = global_orient
#     if body_pose is not None:
#         label_arr[..., base_idx+3:72+base_idx] = body_pose
#     if betas is not None:
#         label_arr[..., base_idx+72:82+base_idx] = betas
#     return label_arr
class AABB(torch.nn.Module):
    def __init__(self, coord_max, coord_min):
        super().__init__()
        self.register_buffer("coord_max", torch.from_numpy(coord_max).float())
        self.register_buffer("coord_min", torch.from_numpy(coord_min).float())

    def normalize(self, x, sym=False):
        # print('self.coord_min', self.coord_min)
        # print('self.coord_max', self.coord_max)
        # print('x', x)
        x = (x - self.coord_min.to(x.device)) / (self.coord_max.to(x.device) - self.coord_min.to(x.device))
        if sym:
            x = 2 * x - 1.
        return x

    def unnormalize(self, x, sym=False):
        if sym:
            x = 0.5 * (x + 1)
        x = x * (self.coord_max - self.coord_min) + self.coord_min
        return x

    def clip(self, x):
        return x.clip(min=self.coord_min, max=self.coord_max)

    def volume_scale(self):
        return self.coord_max - self.coord_min

    def scale(self):
        return math.sqrt((self.volume_scale() ** 2).sum() / 3.)


def getProjectionMatrixTensor(znear, zfar, fovX, fovY, shiftX=0.0, shiftY=0.0):
    """Returns [..., 4,4] projection matrix from view frustum parameters.
    Args:
        znear: [...] Tensor
        zfar: [...]
        fovX: [...]
        fovY: [...]
        shiftX: [...]
        shiftY:  [...]
    Returns:
        [..., 4, 4] Tensor
    """
    # Calculate the tangent values using PyTorch's operations to handle tensors.
    tanHalfFovY = torch.tan((fovY / 2))
    tanHalfFovX = torch.tan((fovX / 2))

    # Calculate frustum boundaries. The operations are automatically broadcasted.
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    # Create a tensor for the projection matrix. We need to ensure it has the correct shape
    # based on the input tensors. Here, we are creating a zero tensor of the right shape.
    # The shape is derived from the input tensors, with an added 4x4 at the end for the matrix itself.
    matrix_shape = znear.shape + (4, 4)
    P = torch.zeros(matrix_shape, dtype=znear.dtype, device=znear.device)

    z_sign = 1.0

    # Fill in the values. We are using PyTorch's operations to ensure compatibility with tensors.
    # The ellipsis is used to automatically handle the arbitrary number of dimensions.
    P[..., 0, 0] = 2.0 * znear / (right - left)
    P[..., 1, 1] = 2.0 * znear / (top - bottom)
    P[..., 0, 2] = (right + left) / (
            right - left
    ) + shiftX  # modified based on the assumed intention
    P[..., 1, 2] = (top + bottom) / (
            top - bottom
    ) + shiftY  # modified based on the assumed intention
    P[..., 3, 2] = z_sign
    P[..., 2, 2] = z_sign * zfar / (zfar - znear)
    P[..., 2, 3] = -(zfar * znear) / (zfar - znear)

    return P


def parse_raw_labels(label_arr: torch.Tensor, scale=1.0):
    smpl_params = label_arr[..., 25:]
    scale = smpl_params[..., 0]
    # print('scale', scale)
    cam2world = label_arr[..., :16].reshape(*label_arr.shape[:-1], 4, 4)
    # print('cam2world', cam2world)
    # print('cam2world.shape', cam2world.shape)  # [2, 4, 4]
    # print('cam2world', cam2world)
    cam2world[..., :3, 3] /= scale
    T = torch.inverse(cam2world)[..., :3, 3]
    intrinsics = label_arr[..., 16:25]
    transl = smpl_params[..., 1:4]
    global_orient = smpl_params[..., 4:7]
    body_pose = smpl_params[..., 7:76]
    betas = smpl_params[..., 76:86]
    # Change intrinsics to range [-1, 1]
    intrinsics = intrinsics.reshape(*label_arr.shape[:-1], 3, 3)
    # intrinsics[0, [0, 1], [0, 1]] *= 2
    fovx = 2 * torch.arctan2(torch.tensor(1.0), 2 * intrinsics[..., 0, 0])
    fovy = 2 * torch.arctan2(torch.tensor(1.0), 2 * intrinsics[..., 1, 1])
    return dict(cam2world=cam2world,
                T=T,
                intrinsics=intrinsics,
                fovy=fovy,
                fovx=fovx,
                principle_points=intrinsics[..., :2, 2],
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas,
                transl=transl
                )


def create_new_camera(parsed_labels, image_width, img_height, device):
    """Create new camera(s) from parsed labels."""

    # Calculations should handle any shape of input. We avoid squeezing or unsqueezing
    # which assumes specific dimensions.
    world_view_transform = torch.inverse(parsed_labels["cam2world"])

    # Using ellipsis to handle broadcasting with any leading dimensions
    shiftX = 2 * parsed_labels["principle_points"][..., 0] - 1
    shiftY = 2 * parsed_labels["principle_points"][..., 1] - 1

    # We assume getProjectionMatrix is capable of handling broadcasting
    proj_matrix = getProjectionMatrixTensor(
        znear=torch.tensor(0.01, device=device).expand_as(shiftX),
        zfar=torch.tensor(100, device=device).expand_as(shiftX),
        fovX=parsed_labels["fovx"],
        fovY=parsed_labels["fovy"],
        shiftX=shiftX, shiftY=shiftY
    ).to(device)

    # We're using the new function 'permute_last_dims' here instead of directly calling 'permute'.
    world_view_transform_permuted = torch.transpose(world_view_transform, -1, -2).to(device)
    full_proj_transform_permuted = torch.transpose(proj_matrix @ world_view_transform, -1, -2).to(device)
    proj_matrix_permuted = torch.transpose(proj_matrix, -1, -2).to(device)

    # print('world_view_transform.inverse().shape', world_view_transform.inverse().shape)

    # camera_center = world_view_transform.inverse()[:, 3, :3]

    # Note: the constructor of MiniCam should also support inputs with arbitrary shapes
    cam = MiniCam(
        width=image_width, height=img_height,
        fovx=parsed_labels["fovx"], fovy=parsed_labels["fovy"],
        znear=0.01,
        zfar=100,
        world_view_transform=world_view_transform_permuted,
        full_proj_transform=full_proj_transform_permuted
    )

    # Set the projection matrix with correct permutation
    cam.projection_matrix = proj_matrix_permuted

    return cam


def get_basedata(c, device):
    body_model = SMPL('/data/vdb/zhongyuhe/workshop/AG3D/training/deformers/smplx/SMPLX', gender='neutral')
    faces = np.load('/data/vde/zhongyuhe/workshop/3dgs-avatar/body_models/misc/faces.npz')['faces']
    # smpl_param = c[:, 25:]
    # cam_param = c[:, :25]
    smpl_param = c[25:].to(device)
    cam_param = c[:25].to(device)
    # device = smpl_param.device
    # print('smpl_param.device', smpl_param.device)

    global_orient = smpl_param[4:7].reshape(-1, 3)

    body_pose = smpl_param[7:76].reshape(-1, 69)
    full_pose = torch.cat((global_orient, body_pose), dim=1)
    # print('body_pose.shape', body_pose.shape)
    transl = smpl_param[1:4].reshape(-1, 3)
    betas = smpl_param[76:].reshape(-1, 10)
    scale = smpl_param[0].reshape(-1, 1)

    # Get shape vertices
    body = body_model(betas=betas, global_orient=torch.zeros_like(global_orient).to(device),
                      body_pose=torch.zeros_like(body_pose).to(device), transl=torch.zeros_like(transl).to(device))
    # minimal_shape = body['vertices'].detach().cpu().numpy()  # template mesh
    minimal_shape = body['vertices'].to(device)

    # Get bone transforms
    body = body_model(global_orient=global_orient, body_pose=body_pose,
                      betas=betas, transl=transl, scale=scale)

    # Visualize SMPL mesh
    vertices = body['vertices'].detach().cpu().numpy()
    pose_hand = body_pose[63:]
    # bone_transforms = body.A.detach().cpu().numpy()  # 对应为A
    bone_transforms = body.A.to(device)
    # print('bone_transforms.shape', bone_transforms.shape)  # 应为 (24, 4, 4)
    # Jtr_posed = body.joints.detach().cpu().numpy()
    Jtr_posed = body.joints.to(device)


    return {'minimal_shape': minimal_shape,
            'betas': betas,
            'full_pose': full_pose,
            'Jtr_posed': Jtr_posed,
            'bone_transforms': bone_transforms,
            'transl': transl,
            'global_orient': global_orient,
            'body_pose': body_pose,
            'pose_hand': pose_hand}


def get_02v_bone_transforms(Jtr, ):
    rot45p = Rotation.from_euler('z', 45, degrees=True).as_matrix()
    rot45n = Rotation.from_euler('z', -45, degrees=True).as_matrix()

    # Specify the bone transformations that transform a SMPL A-pose mesh
    # to a star-shaped A-pose (i.e. Vitruvian A-pose)
    bone_transforms_02v = np.tile(np.eye(4), (24, 1, 1))

    # First chain: L-hip (1), L-knee (4), L-ankle (7), L-foot (10)
    chain = [1, 4, 7, 10]
    rot = rot45p.copy()
    for i, j_idx in enumerate(chain):
        bone_transforms_02v[j_idx, :3, :3] = rot
        t = Jtr[j_idx].copy()
        if i > 0:
            parent = chain[i - 1]
            t_p = Jtr[parent].copy()
            t = np.dot(rot, t - t_p)
            t += bone_transforms_02v[parent, :3, -1].copy()

        bone_transforms_02v[j_idx, :3, -1] = t

    bone_transforms_02v[chain, :3, -1] -= np.dot(Jtr[chain], rot.T)
    # Second chain: R-hip (2), R-knee (5), R-ankle (8), R-foot (11)
    chain = [2, 5, 8, 11]
    rot = rot45n.copy()
    for i, j_idx in enumerate(chain):
        bone_transforms_02v[j_idx, :3, :3] = rot
        t = Jtr[j_idx].copy()
        if i > 0:
            parent = chain[i - 1]
            t_p = Jtr[parent].copy()
            t = np.dot(rot, t - t_p)
            t += bone_transforms_02v[parent, :3, -1].copy()

        bone_transforms_02v[j_idx, :3, -1] = t

    bone_transforms_02v[chain, :3, -1] -= np.dot(Jtr[chain], rot.T)

    return bone_transforms_02v


def get_metadata(c, device):
    faces = np.load('body_models/misc/faces.npz')['faces']
    skinning_weights = dict(np.load('body_models/misc/skinning_weights_all.npz'))
    posedirs = dict(np.load('body_models/misc/posedirs_all.npz'))
    J_regressor = dict(np.load('body_models/misc/J_regressors.npz'))
    gender = 'neutral'

    # minimal_shape, betas, Jtr_posed, bone_transforms, transl, global_orient, body_pose, pose_hand = get_basedata(
    #     c)
    basedata = get_basedata(c, device)
    # minimal_shape = basedata['minimal_shape'].astype(np.float32).reshape(6890, 3)  # 感觉只能处理batch_size=1这种情况
    minimal_shape = basedata['minimal_shape'].reshape(6890, 3).detach().cpu().numpy()

    J_regressor = J_regressor[gender]
    Jtr = np.dot(J_regressor, minimal_shape)
    # print('J_regressor.shape', J_regressor.shape)  # [24, 6890]
    # print('minimal_shape.shape', minimal_shape.shape)  # [2, 6890, 3]
    skinning_weights = skinning_weights[gender]

    pose = Rotation.from_rotvec(basedata['full_pose'].cpu().reshape([-1, 3]))
    pose_mat_full = pose.as_matrix()  # 24 x 3 x 3
    pose_mat = pose_mat_full[1:, ...].copy()  # 23 x 3 x 3

    # pose_rot = torch.cat((torch.eye(3).unsqueeze(0), pose_mat), dim=0).reshape(-1, 9)  # 24 x 9, root rotation is set to identity
    pose_rot = torch.from_numpy(np.concatenate([np.expand_dims(np.eye(3), axis=0), pose_mat], axis=0).reshape(
        [-1, 9])).to(device)
    # rots = torch.from_numpy(pose_rot).float().unsqueeze(0)

    bone_transforms = basedata['bone_transforms'].detach().cpu().numpy()  # (1, 24, 4, 4)
    bone_transforms_02v = get_02v_bone_transforms(Jtr)
    bone_transforms = bone_transforms @ np.linalg.inv(bone_transforms_02v)
    bone_transforms = bone_transforms.astype(np.float32)
    # print('bone_transforms.shape', bone_transforms.shape)  #
    # print('basedata[transl].shape', basedata['transl'].shape)  # [1, 3]
    bone_transforms[:, :, :3, 3] += basedata['transl'].cpu().numpy()  # add global offset
    bone_transforms = torch.from_numpy(bone_transforms).to(device)

    T = np.matmul(skinning_weights, bone_transforms_02v.reshape([-1, 16])).reshape([-1, 4, 4])
    vertices = np.matmul(T[:, :3, :3], minimal_shape[..., np.newaxis]).squeeze(-1) + T[:, :3, -1]

    center = np.mean(minimal_shape, axis=0)
    minimal_shape_centered = minimal_shape - center
    cano_max = minimal_shape_centered.max()
    cano_min = minimal_shape_centered.min()
    padding = (cano_max - cano_min) * 0.05
    # compute pose condition
    Jtr_norm = Jtr - center
    Jtr_norm = (Jtr_norm - cano_min + padding) / (cano_max - cano_min) / 1.1
    Jtr_norm -= 0.5
    Jtr_norm *= 2.
    Jtr_norm = torch.from_numpy(Jtr_norm).to(device)

    coord_max = np.max(vertices, axis=0)
    coord_min = np.min(vertices, axis=0)

    # padding_ratio = self.cfg.padding
    padding_ratio = 0.1
    padding_ratio = np.array(padding_ratio, dtype=np.float)
    padding = (coord_max - coord_min) * padding_ratio
    coord_max += padding
    coord_min -= padding

    cano_mesh = trimesh.Trimesh(vertices=vertices.astype(np.float32), faces=faces)
    aabb = AABB(coord_max, coord_min)
    # print('coord_max', coord_max)
    # print('coord_min', coord_min)

    pcd = readPointCloud(vertices, faces)  # canonical mesh
    # print('Jtr_norm.device', Jtr_norm.device)
    # print('pose_rot.device', pose_rot.device)

    return {
        'faces': faces,
        'posedirs': posedirs,
        'J_regressor': J_regressor,
        'cameras_extent': 3.469298553466797,
        # hardcoded, used to scale the threshold for scaling/image-space gradient
        # 'frame_dict': frame_dict,
        'gender': gender,
        'smpl_verts': vertices.astype(np.float32),
        'minimal_shape': minimal_shape,
        'Jtrs': Jtr_norm.unsqueeze(0).to(device),
        'rots': pose_rot.unsqueeze(0).to(device),
        # 'skinning_weights': skinning_weights.astype(np.float32),
        'skinning_weights': torch.from_numpy(skinning_weights).to(device),
        'bone_transforms': bone_transforms,
        'cano_mesh': cano_mesh,
        'pcd': pcd,

        'coord_min': torch.from_numpy(coord_min).to(device),
        'coord_max': torch.from_numpy(coord_max).to(device),
        'aabb': aabb,
    }
    # metadata.update(cano_data)
    #     if self.cfg.train_smpl:  # default = false
    #         self.metadata.update(self.get_smpl_data())

def readPointCloud(vertices, faces):
        ply_path = '/data/vde/zhongyuhe/workshop/mycode/3DGS_Gen/out/human/cano.ply'
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        n_points = 50_000
        xyz = mesh.sample(n_points)
        rgb = np.ones_like(xyz) * 255
        storePly(ply_path, xyz, rgb)
        pcd = fetchPly(ply_path)
        return pcd

def get_bg_color(bg_color:str):
    """Range between 0, 1"""
    if bg_color == 'white':
        background = torch.ones((3,), dtype=torch.float32)
    elif bg_color == "gray":
        background = 0.5 * torch.ones((3,), dtype=torch.float32)
    elif bg_color == "black":
        background = torch.zeros((3,), dtype=torch.float32)
    elif bg_color == "random":
        background = torch.rand((3,), dtype=torch.float32)
    else:
        raise ValueError("Invalid Color!")
    return background

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def save_gaussians_to_ply(gaussians, path):
    # mkdir_p(os.path.dirname(path))
    GaussianModelMini().from_another(gaussians).save_ply(path)