import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.spatial.transform import Rotation as R

# https://github.com/erikwijmans/Pointnet2_PyTorch
from pointnet2_ops.pointnet2_utils import furthest_point_sample
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule, PointnetSAModuleMSG
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG
from pointnet2.models.pointnet2_ssg_sem import PointNet2SemSegSSG

class AffordanceMSG(PointNet2SemSegSSG):

    def __init__(self, hparams):
        super().__init__(hparams)

        self.hparams = hparams
        self.sigmoid = nn.Sigmoid()

        self._build_model()

    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        c_in = 3 if "feat_dim" not in self.hparams.keys() else self.hparams["feat_dim"]
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=1024,
                radii=[0.05, 0.1],
                nsamples=[16, 32],
                mlps=[[c_in, 16, 16, 32], [c_in, 32, 32, 64]],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )
        c_out_0 = 32 + 64

        c_in = c_out_0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=256,
                radii=[0.1, 0.2],
                nsamples=[16, 32],
                mlps=[[c_in, 64, 64, 128], [c_in, 64, 96, 128]],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )
        c_out_1 = 128 + 128

        c_in = c_out_1
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=64,
                radii=[0.2, 0.4],
                nsamples=[16, 32],
                mlps=[[c_in, 128, 196, 256], [c_in, 128, 196, 256]],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )
        c_out_2 = 256 + 256

        c_in = c_out_2
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=16,
                radii=[0.4, 0.8],
                nsamples=[16, 32],
                mlps=[[c_in, 256, 256, 512], [c_in, 256, 384, 512]],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )
        c_out_3 = 512 + 512

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 6, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_0, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_1, 512, 512]))
        self.FP_modules.append(PointnetFPModule(mlp=[c_out_3 + c_out_2, 512, 512]))

        self.fc_lyaer = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, 1, kernel_size=1),
        )

    def get_loss(self, pcs, affords):
        pcs = pcs.repeat(1, 1, 2)
        affords_pred = self.forward(pcs)

        # affordance_min = torch.unsqueeze(torch.min(affords_pred, dim=2).values, 1)
        # affordance_max = torch.unsqueeze(torch.max(affords_pred, dim=2).values, 1)
        # affords_pred = (affords_pred - affordance_min) / (affordance_max - affordance_min)

        BCE_loss = F.binary_cross_entropy_with_logits(affords_pred, affords.unsqueeze(1))
        return BCE_loss

    def inference(self, pcs):
        pcs = pcs.repeat(1, 1, 2)
        affords_pred = self.forward(pcs)

        return affords_pred

class Affordance(PointNet2ClassificationSSG):

    def __init__(self, hparams):
        super().__init__(hparams)

        self.hparams = hparams
        # self.loss = nn.MSELoss(reduction='mean')
        # self.loss = F.binary_cross_entropy_with_logits()
        self.sigmoid = nn.Sigmoid()

        self._build_model()

    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=1024,
                radius=0.1,
                nsample=32,
                mlp=[3, 32, 32, 64],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[64, 64, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=16,
                radius=0.8,
                nsample=32,
                mlp=[256, 256, 256, 512],
                use_xyz=True,
            )
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128 + 3, 128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 64, 256, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 256, 256]))

        self.fc_layer = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, 1, kernel_size=1),
            # nn.Sigmoid()
        )
    
    def _break_up_pc(self, pc):
        xyz = pc[..., :3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud, return_feat=False):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        if return_feat:
            return self.fc_layer(l_features[0]), l_features[0]
        return self.fc_layer(l_features[0])

    def get_loss(self, pcs, affords):
        pcs = pcs.repeat(1, 1, 2)
        affords_pred = self.forward(pcs)
        
        # affordance_min = torch.unsqueeze(torch.min(affords_pred, dim=2).values, 1)
        # affordance_max = torch.unsqueeze(torch.max(affords_pred, dim=2).values, 1)
        # affords_pred = (affords_pred - affordance_min) / (affordance_max - affordance_min)

        BCE_loss = F.binary_cross_entropy_with_logits(affords_pred, affords.unsqueeze(1))
        return BCE_loss

    def inference(self, pcs, return_feat=False):
        pcs = pcs.repeat(1, 1, 2)
        
        if return_feat:
            affords_pred, feat = self.forward(pcs, return_feat)
        else :
            affords_pred = self.forward(pcs)

        if return_feat:
            return affords_pred, feat
        return affords_pred

    def inference_sigmoid(self, pcs, return_feat=False):
        pcs = pcs.repeat(1, 1, 2)
    
        if return_feat:
            affords_pred, feat = self.forward(pcs, return_feat)
        else :
            affords_pred = self.forward(pcs)

        if return_feat:
            return self.sigmoid(affords_pred), feat
        return self.sigmoid(affords_pred)
