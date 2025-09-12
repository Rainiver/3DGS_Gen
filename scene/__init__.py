#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from models import GaussianConverter
from scene.gaussian_model import GaussianModel


class Scene:
    gaussians: GaussianModel
    def __init__(self, gaussians: GaussianModel, data):
        self.gaussians = gaussians
        self.metadata = data
        self.gaussians.create_from_pcd(self.metadata['pcd'],
                                       spatial_lr_scale=self.metadata['cameras_extent'])  # 从点云得到各个高斯属性
        # self.gaussians.save_ply('./out/init.ply')
        self.converter = GaussianConverter(self.metadata).cuda()
    def convert_gaussians(self, camera, data):
        return self.converter(self.gaussians, camera, data)

