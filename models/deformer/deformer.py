import torch.nn as nn

# from models.deformer.rigid import get_rigid_deform, SkinningField
from models.deformer.rigid import SkinningField
# from models.deformer.non_rigid import get_non_rigid_deform, HashGridwithMLP
from models.deformer.non_rigid import HashGridwithMLP

class Deformer(nn.Module):
    # def __init__(self, cfg, metadata):
    def __init__(self, metadata):
        super().__init__()
        # self.cfg = cfg
        # self.rigid = get_rigid_deform(cfg.rigid, metadata)  # skinning_field
        self.rigid = SkinningField(metadata)
        # self.non_rigid = get_non_rigid_deform(cfg.non_rigid, metadata)  # hashgrid
        self.non_rigid = HashGridwithMLP(metadata)

    # def forward(self, gaussians, camera, iteration, compute_loss=True):
    def forward(self, gaussians, data):
        loss_reg = {}
        # deformed_gaussians, loss_non_rigid = self.non_rigid(gaussians, iteration, camera, compute_loss)
        deformed_gaussians = self.non_rigid(gaussians, data)
        deformed_gaussians.save_ply('./out/non_rigid_deformed_gs.ply')
        # deformed_gaussians = self.rigid(deformed_gaussians, iteration, camera)
        deformed_gaussians = self.rigid(deformed_gaussians, data)
        deformed_gaussians.save_ply('./out/rigid_deformed_gs.ply')

        # loss_reg.update(loss_non_rigid)  # 高斯如何正则化？需要引入吗
        # return deformed_gaussians, loss_reg
        return deformed_gaussians

# def get_deformer(cfg, metadata):
#     return Deformer(cfg, metadata)
def get_deformer(metadata):
    return Deformer(metadata)
