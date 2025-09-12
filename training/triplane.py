# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Modified by Zijian Dong for AG3D: Learning to Generate 3D Avatars from 2D Image Collections

"""Generator architectures from the paper
"AG3D: Learning to Generate 3D Avatars from 2D Image Collections"

Code adapted from
"Efficient Geometry-aware 3D Generative Adversarial Networks."""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_utils import persistence
from training.networks_stylegan2 import Generator as StyleGAN2Backbone
# from training.volumetric_rendering.renderer import ImportanceRenderer
# from training.volumetric_rendering.ray_sampler import RaySampler
from utils.data_utils import create_new_camera, parse_raw_labels, get_metadata
from gaussian_renderer import render
from scene import Scene, GaussianModel
import dnnlib
import numpy as np
from tqdm import tqdm
import trimesh
from training.networks_stylegan2 import FullyConnectedLayer
from utils.data_utils import get_bg_color


class PipelineParams():
    def __init__(self):
        self.convert_SHs_python = True
        self.compute_cov3D_python = True
        self.debug = False

@persistence.persistent_class
# class AG3DGenerator(torch.nn.Module):
class GSGenerator(torch.nn.Module):

    def __init__(self,
                 z_dim,  # Input latent (Z) dimensionality.
                 c_dim,  # Conditioning label (C) dimensionality.
                 w_dim,  # Intermediate latent (W) dimensionality.
                 img_resolution,  # Output resolution.
                 img_channels,  # Number of output color channels.
                 sr_num_fp16_res=0,
                 is_sr_module=False,
                 bg_color='white',
                 mapping_kwargs={},  # Arguments for MappingNetwork.
                 rendering_kwargs={},
                 sr_kwargs={},
                 **synthesis_kwargs,  # Arguments for SynthesisNetwork.
                 ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        # self.renderer = ImportanceRenderer()
        # self.ray_sampler = RaySampler()
        self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=256, img_channels=32 * 3,
                                          mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        self.decoder = OSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1),
                                       'decoder_output_dim': 32})
        self.superresolution = dnnlib.util.construct_class_by_name(
            class_name=rendering_kwargs['superresolution_module'], channels=32, img_resolution=img_resolution,
            sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)

        self.neural_rendering_resolution = img_resolution // 2

        self.rendering_kwargs = rendering_kwargs

        self._last_planes = None
        self.gaussians = GaussianModel(
            use_sh=False,
            sh_degree=3,
            delay=1000,
            feature_dim=32,
        )

        self.pipeline = PipelineParams()
        self.background = get_bg_color(bg_color)

    def parse_raw_labels(self, c):
        # if self._template_type == "smpl":
        return parse_raw_labels(c)

    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):

        if self.rendering_kwargs['c_gen_conditioning_zero']:
            c = torch.zeros_like(c)
        return self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi,
                                     truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def synthesis(self, ws, c, neural_rendering_resolution=None, update_emas=False, cache_backbone=False,
                  use_cached_backbone=False, patch_params=None, cano=False, **synthesis_kwargs):

        # cam2world_matrix = c[:, :16].view(-1, 4, 4)
        # intrinsics = c[:, 16:25].view(-1, 3, 3)
        # print('cam2world_matrix.shape', cam2world_matrix.shape)
        # print('intrinsics', intrinsics.shape)
        # print('c.shape', c.shape)
        # camera = c[:, :25]
        #
        # if not cano:
        #     smpl_params = {
        #     'betas': c[:,101:], # 10 betas
        #     'body_pose': c[:,32:101], # 69 rot-> 23*3
        #     'global_orient': c[:,29:32], # 3 rot
        #     'transl': c[:,26:29], # 3 transl
        #     'scale':c[:, 25]
        #     }
        # else:
        #     smpl_params = None
        #
        # if neural_rendering_resolution is None:
        #     neural_rendering_resolution = self.neural_rendering_resolution
        # else:
        #     self.neural_rendering_resolution = neural_rendering_resolution

        # print('cam2world_matrix', cam2world_matrix.shape)
        # print(cam2world_matrix)
        # print('intrinsics', intrinsics.shape)
        # print(intrinsics)

        # Create a batch of rays for volume rendering
        # ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution, patch_params=patch_params)

        # print('ray_origins', ray_origins.shape)
        # print(ray_origins)
        # print('ray_directions', ray_directions.shape)
        # print(ray_directions)

        # Create triplanes by running StyleGAN backbone
        # N, M, _ = ray_origins.shape
        # print('N', N)
        c_clone = c.clone().detach()

        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
            # color_, opacity_, scaling_, rotation_ = self.backbone.synthesis(ws, update_emas=update_emas, lr_mul=lr_mul,
            #                                                                 **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        conv = nn.Conv2d(96, 11, kernel_size=1, stride=1)
        # input_reshaped = planes.view(planes.shape[0], 96, -1).detach().cpu()# shape变为 [1, 96, 65536]
        input_reshaped = planes.detach().cpu()
        output = conv(input_reshaped).to(planes.device)  # shape现在是 [1, 11, 65536]
        # 进行变换还原为原本的spatial尺寸
        feature_gs = output.view(ws.shape[0], 11, planes.shape[2], planes.shape[3])  # shape变回 [1, 11, 256, 256]
        # print('feature_gs', feature_gs)
        # print('feature_gs.shape', feature_gs.shape)  # [4, 11, 256, 256]
        # feature_gs = output_reshaped.permute(0, 2, 3, 1)  # shape变为 [1, 256, 256, 11]
        # 在范围 0-256 中生成一个形状为 [50000, 2] 的均匀随机样本，同时也将其转换为 float32 以便兼容插值函数
        torch.manual_seed(0)
        # 生成50000个点的坐标，每个点的坐标为2维(0到255之间，包括0但不包括256)
        coords = torch.rand(50000, 2) * 255

        # 对坐标进行规范化，将其限制在-1到1之间，这是grid_sample期望的格式
        # grid_sample中的坐标(-1, -1)分别表示图片的左上角， (1, 1)表示图片的右下角
        coords = coords / 255  # 将坐标规范化到0到1之间
        coords = coords * 2 - 1  # 将坐标规范化到-1到1之间
        coords = coords.unsqueeze(0).unsqueeze(0).to(planes.device)  # 将坐标变形以满足grid_sample的期望输入形状
        batch_size = feature_gs.size(0)
        coords = coords.expand(batch_size, -1, -1, -1)

        # 使用双线性插值进行采样，coords需要是 N x H x W x 2形状
        # 我们使用unsqueeze给coords添加了两个额外的维度，所以它现在是 1 x 1 x 50000 x 2 形状
        # 这会将coords扩展为与batch中的每张图片对齐
        sampled = F.grid_sample(feature_gs, coords, mode='bilinear', align_corners=True)

        # sampled 的形状是 [4, 11, 1, 50000]，因为我们使用了一个 1 x 1 x 50000网格
        # 对四张图片中的每一张都进行了采样
        # print('sampled', sampled)
        # print('sampled.shape', sampled.shape)

        # 变换输出形状以移除它的单独Height和Width维度
        out_feature = sampled.squeeze(2)  # 移除H维度，得到的形状为[4, 11, 50000]
        # print('out_feature.shape', out_feature.shape)
        out_feature = out_feature.permute(0, 2, 1).to(planes.device)
        # coords = torch.rand(ws.shape[0], 50000, 2).to(planes.device) * 256
        # coords = coords.float()
        # # 将张量重塑为形状 [1, 1, num_samples, 2]
        # # coords = coords.unsqueeze(0).unsqueeze(0)
        # coords = coords.unsqueeze(1)
        # # 执行网格采样
        # output = F.grid_sample(feature_gs, coords, mode='bilinear', padding_mode='border', align_corners=True)
        # # 重塑输出
        # out_feature = output.view(feature_gs.shape[0], 50000, 11).to(planes.device)
        # print('out_feature', out_feature)

        rendering = []

        for bs in range(len(planes)):
            # print('planes.device', planes.device)
            # 把c按照batch_size切分
            # print('c[bs].device', c_clone[bs].device)
            labels = self.parse_raw_labels(c_clone[bs])
            camera = create_new_camera(labels, self.img_resolution, self.img_resolution, planes.device)
            # print('camera.camera_center', camera.camera_center)
            data = get_metadata(c_clone[bs], planes.device)
            # print('data.device', data.device)
            self.gaussians = GaussianModel(
                use_sh=False,
                sh_degree=3,
                delay=1000,
                feature_dim=32
            )
            scene = Scene(self.gaussians, data)

            # print('planes.shape', planes.shape)  # [1, 96, 256, 256]
        # fc = FullyConnectedLayer(in_features=96, out_features=11)
            self.gaussians.set_features_dc(out_feature[bs, ..., :3])  # 3维
            self.gaussians.set_opacity(out_feature[bs, ..., 3:4])  # 1维
            self.gaussians.set_scaling(out_feature[bs, ..., 4:7])  # 3维
            self.gaussians.set_rotation(out_feature[bs, ..., 7:])  # 4维

            # print('self.gaussians._features_dc_triplane', self.gaussians._features_dc)

            self.gaussians.save_ply('./out/gen_0.ply')

            render_pkg = render(camera, data, scene, self.pipeline, self.background.to(planes.device), device=planes.device)
            # render_pkg = render(camera, data, scene, self.pipeline)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
            render_pkg["visibility_filter"], render_pkg["radii"]
        # opacity = render_pkg["opacity_render"] if use_mask else None
            rendering.append(image)

        rendering = torch.stack(rendering)
        # print('rendering.shape', rendering.shape)  # [2, 3, 256, 256]

        # print('rendering_kwargs', self.rendering_kwargs)

        # print('feature_samples.shape', feature_samples.shape)  # [1, 65536, 32]
        # print('weights_samples.shape', weights_samples.shape)  # [1, 65536, 1]
        # print('feature_samples', (feature_samples.reshape(1, 256, 256, 32))[:, 120:130, 120:130, :3])

        # Reshape into 'raw' neural-rendered image
        # H = W = neural_rendering_resolution

        # permute[1, 32 , 65536].reshape(1, 32 , 256, 256)
        # feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W)

        # depth_image[1, 1, 256, 256]
        # depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # if self.rendering_kwargs['is_normal']:
        #     grad_image = grad_samples.permute(0, 2, 1).reshape(N, grad_samples.shape[-1], H, W)
        # else:
        #     grad_image = None

        # grad_image = None

        # Run superresolution to get final image

        # rgb_image = feature_image[:, :self.img_channels]
        # print('rgb_image', rgb_image.shape)
        # print(rgb_image[:,:,120:130,120:130])

        # sr_image = self.superresolution(rgb_image.clone(), feature_image, ws,
        #                                 noise_mode=self.rendering_kwargs['superresolution_noise_mode'],
        #                                 **{k: synthesis_kwargs[k] for k in synthesis_kwargs.keys() if
        #                                    k != 'noise_mode'})

        # return {'image': sr_image,
        #         'image_raw': rgb_image,
        #         'image_normal': grad_image,
        #         'image_depth': depth_image,
        #         'grad_cano': grad_cano_samples,
        #         'sdf': sdf_samples}
        return {'image': rendering,
                'image_raw': rendering
                }

    def sample(self, coordinates, directions, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False,
               **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
                          update_emas=update_emas)
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def sample_mixed(self, coordinates, directions, ws, truncation_psi=1, truncation_cutoff=None, update_emas=False,
                     smpl_params=None, **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs,
                                       smpl_params=smpl_params)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None,
                update_emas=False, cache_backbone=False, use_cached_backbone=False, patch_params=None, cano=False,
                **synthesis_kwargs):
        # Render a batch of generated images.
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
                          update_emas=update_emas)

        return self.synthesis(ws, c, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution,
                              cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone,
                              patch_params=patch_params, cano=cano, **synthesis_kwargs)

    def get_mesh(self, z, c, ws=None, voxel_resolution=256, truncation_psi=1, truncation_cutoff=None, update_emas=False,
                 canonical=False, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        device = z.device

        if not canonical:
            smpl_params = {
                'betas': c[:, 101:],
                'body_pose': c[:, 32:101],
                'global_orient': c[:, 29:32],
                'transl': c[:, 26:29],
                'scale': c[:, 25]
            }
        else:
            smpl_params = {
                'betas': c[:, 101:] * 0,
                'body_pose': c[:, 32:101] * 0,
                'global_orient': c[:, 29:32] * 0,
                'transl': c[:, 26:29] * 0,
                'scale': c[:, 25] * 0 + 1
            }
            smpl_params['transl'][:, 1] = 0.3

        if ws is None:
            ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
                              update_emas=update_emas)
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])

        # print(smpl_params)
        smpl_outputs = self.renderer.deformer.body_model(betas=smpl_params["betas"],
                                                         body_pose=smpl_params["body_pose"],
                                                         global_orient=smpl_params["global_orient"],
                                                         transl=smpl_params["transl"],
                                                         scale=smpl_params["scale"])

        face = self.renderer.deformer.smpl_faces
        smpl_verts = smpl_outputs.vertices.float()[0]
        # # vis smpl
        # pts = smpl_verts.data.cpu().numpy()
        # pcd = trimesh.PointCloud(pts)
        # pcd.export('smpl_verts.ply')
        # print('save smpl_verts.ply')

        scale = 1.1  # Scale of the padded bbox regarding the tight one.
        verts = smpl_verts.data.cpu().numpy()
        gt_bbox = np.stack([verts.min(axis=0), verts.max(axis=0)], axis=0)
        gt_center = (gt_bbox[0] + gt_bbox[1]) * 0.5
        gt_scale = (gt_bbox[1] - gt_bbox[0]).max()

        samples, voxel_origin, voxel_size = create_samples(N=voxel_resolution, smpl_verts=smpl_verts)
        samples = samples.to(device)
        # print('samples: ', samples.shape)

        # min and max value of samples x
        # print('min_x: ', samples[:,:,0].min())
        # print('max_x: ', samples[:,:,0].max())
        # min and max value of samples y
        # print('min_y: ', samples[:,:,1].min())
        # print('max_y: ', samples[:,:,1].max())
        # min and max value of samples z
        # print('min_z: ', samples[:,:,2].min())
        # print('max_z: ', samples[:,:,2].max())

        sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=device)

        head = 0
        max_batch = int(1e7)
        with tqdm(total=samples.shape[1]) as pbar:
            with torch.no_grad():
                while head < samples.shape[1]:
                    torch.manual_seed(0)
                    sigma = self.sample_mixed(samples[:, head:head + max_batch], samples[:, head:head + max_batch], ws,
                                              truncation_psi=truncation_psi, noise_mode='const',
                                              smpl_params=smpl_params)['sdf']
                    sigmas[:, head:head + max_batch] = sigma
                    head += max_batch
                    pbar.update(max_batch)

        sigmas = sigmas.reshape((voxel_resolution, voxel_resolution, voxel_resolution)).cpu().numpy()

        # import skimage.measure
        # import os
        # verts, faces, normals, values = skimage.measure.marching_cubes(sigmas, level=10, spacing=[1] * 3)
        # box_warp = 1.9
        # verts = verts * box_warp / voxel_resolution - 1
        #
        # mesh = trimesh.Trimesh(verts, faces[..., ::-1])
        # # outpath = os.path.join(outdir, f'seed{seed:04d}.ply')
        # # mesh.export(outpath)
        # # mesh.export(outpath.replace('.ply', '.obj'))
        # # print('mesh save to %s' % outpath)
        # verts_torch = torch.from_numpy(verts).float().to(device).unsqueeze(0)

        sigmas = np.flip(sigmas, 0)
        from utils.shape_utils import convert_sdf_samples_to_ply
        mesh = convert_sdf_samples_to_ply(sigmas, [0, 0, 0], 1, level=0)

        verts = mesh.vertices
        verts = (verts / voxel_resolution - 0.5) * scale
        # flip along x axis
        verts[:, 0] = -verts[:, 0]

        verts = verts * gt_scale + gt_center

        verts_torch = torch.from_numpy(verts).float().to(device).unsqueeze(0)

        mesh = trimesh.Trimesh(vertices=verts, faces=mesh.faces, process=False)

        weights = self.renderer.deformer.deformer.query_weights(verts_torch).clamp(0, 1)[0]

        mesh.visual.vertex_colors[:, :3] = weights2colors(weights.data.cpu().numpy() * 0.999) * 255
        return mesh, weights


class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'],
                                lr_multiplier=options['decoder_lr_mul'])
        )

    def forward(self, sampled_features, ray_directions):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        if sampled_features.ndim == 2:
            # print("---------sampled_features.ndim=2 ------", sampled_features)
            x = sampled_features
        else:
            x = sampled_features.flatten(0, 1)

        x = self.net(x)
        # print("--------------decoder x ------------", x)

        rgb = torch.sigmoid(x[..., 1:]) * (1 + 2 * 0.001) - 0.001  # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma}


def create_samples(N=256, smpl_verts=None, cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    verts = smpl_verts.data.cpu().numpy()
    scale = 1.1
    gt_bbox = np.stack([verts.min(axis=0), verts.max(axis=0)], axis=0)
    gt_center = (gt_bbox[0] + gt_bbox[1]) * 0.5
    gt_scale = (gt_bbox[1] - gt_bbox[0]).max()

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples = (samples / N - 0.5) * scale
    samples = samples * gt_scale + gt_center

    num_samples = N ** 3

    return samples.unsqueeze(0), None, None


def weights2colors(weights):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap('Paired')

    colors = ['pink',  # 0
              'blue',  # 1
              'green',  # 2
              'red',  # 3
              'pink',  # 4
              'pink',  # 5
              'pink',  # 6
              'green',  # 7
              'blue',  # 8
              'red',  # 9
              'pink',  # 10
              'pink',  # 11
              'pink',  # 12
              'blue',  # 13
              'green',  # 14
              'red',  # 15
              'cyan',  # 16
              'darkgreen',  # 17
              'pink',  # 18
              'pink',  # 19
              'blue',  # 20
              'green',  # 21
              'pink',  # 22
              'pink'  # 23
              ]

    color_mapping = {'cyan': cmap.colors[3],
                     'blue': cmap.colors[1],
                     'darkgreen': cmap.colors[1],
                     'green': cmap.colors[3],
                     'pink': [1, 1, 1],
                     'red': cmap.colors[5],
                     }

    for i in range(len(colors)):
        colors[i] = np.array(color_mapping[colors[i]])

    colors = np.stack(colors)[None]  # [1x24x3]
    verts_colors = weights[:, :, None] * colors
    verts_colors = verts_colors.sum(1)
    return verts_colors
