import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.spatial.transform import Rotation as R

# https://github.com/erikwijmans/Pointnet2_PyTorch
from pointnet2_ops.pointnet2_utils import furthest_point_sample
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG

# def KL(mu, logvar):
#     mu = mu.view(mu.shape[0], -1)
#     logvar = logvar.view(logvar.shape[0], -1)
#     loss = 0.5 * torch.sum(mu * mu + torch.exp(logvar) - 1 - logvar, 1)
#     # high star implementation
#     # torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1. - z_var, 1))
#     loss = torch.mean(loss)
#     return loss

def KL(mu, logvar):
    kl_loss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    kl_loss = torch.mean(kl_loss)
    return kl_loss

class PointNet2SemSegSSG(PointNet2ClassificationSSG):
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
            nn.Conv1d(128, self.hparams['feat_dim'], kernel_size=1, bias=False),
            nn.BatchNorm1d(self.hparams['feat_dim']),
            nn.ReLU(True),
        )
    
    def _break_up_pc(self, pc):
        xyz = pc[..., :3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
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

        return self.fc_layer(l_features[0])


class TrajEncoder(nn.Module):
    def __init__(self, traj_feat_dim, num_steps=30, wpt_dim=3):

        # traj_feat_dim = 128 for VAT-Mart

        super(TrajEncoder, self).__init__()

        # self.mlp = nn.Sequential(
        #     nn.Linear(num_steps * wpt_dim, 128),
        #     nn.Linear(128, 128),
        #     nn.Linear(128, traj_feat_dim)
        # )

        self.mlp = nn.Sequential(
            nn.Linear(num_steps * wpt_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, traj_feat_dim)
        )

        self.num_steps = num_steps
        self.wpt_dim = wpt_dim

    # pcs_feat B x F, query_fats: B x 6
    # output: B
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.mlp(x.view(batch_size, self.num_steps * self.wpt_dim))
        return x

# CVAE encoder
class AllEncoder(nn.Module):
    def __init__(self, pcd_feat_dim=128, traj_feat_dim=256, cp_feat_dim=32, hidden_dim=128, z_feat_dim=64):
        super(AllEncoder, self).__init__()

        self.mlp1 = nn.Linear(pcd_feat_dim + cp_feat_dim + traj_feat_dim, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, z_feat_dim)
        self.get_mu = nn.Linear(z_feat_dim, z_feat_dim)
        self.get_logvar = nn.Linear(z_feat_dim, z_feat_dim)

        self.z_dim = z_feat_dim

    # pcs_feat B x F, query_fats: B x 6
    # output: B
    def forward(self, pn_feat, traj_feat, cp_feat):
        net = torch.cat([pn_feat, traj_feat, cp_feat], dim=-1)
        net = F.leaky_relu(self.mlp1(net))
        net = self.mlp2(net)
        mu = self.get_mu(net)
        logvar = self.get_logvar(net)
        noise = torch.Tensor(torch.randn(*mu.shape)).cuda()
        z = mu + torch.exp(logvar / 2) * noise
        return z, mu, logvar

# CVAE decoder
class AllDecoder(nn.Module):
    def __init__(self, pcd_feat_dim, cp_feat_dim=32, z_feat_dim=64, hidden_dim=128, num_steps=30, wpt_dim=3):
        super(AllDecoder, self).__init__()

        # self.mlp = nn.Sequential(
        #     nn.Linear(pcd_feat_dim + cp_feat_dim + z_feat_dim, 512),
        #     nn.Linear(512, 256),
        #     nn.Linear(256, num_steps * wpt_dim)
        # )

        self.mlp = nn.Sequential(
            nn.Linear(pcd_feat_dim + cp_feat_dim + z_feat_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, num_steps * wpt_dim)
        )
        
        self.num_steps = num_steps
        self.wpt_dim = wpt_dim

    # pn_feat B x F, query_fats: B x 6
    # output: B
    def forward(self, pn_feat, cp_feat, z_all):
        batch_size = z_all.shape[0]
        x = torch.cat([pn_feat, cp_feat, z_all], dim=-1)
        x = self.mlp(x)
        x = x.view(batch_size, self.num_steps, 3)
        return x

class VATMart3D(nn.Module):
    def __init__(self, pcd_feat_dim=256, traj_feat_dim=128, cp_feat_dim=32,  
                        hidden_dim=128, z_feat_dim=64, 
                        num_steps=30, wpt_dim=3,
                        lbd_kl=1.0, lbd_recon=1.0, lbd_dir=1.0, kl_annealing=0, dataset_type=0):
        super(VATMart3D, self).__init__()

        self.z_dim = z_feat_dim

        self.pointnet2 = PointNet2SemSegSSG({'feat_dim': pcd_feat_dim})
        self.mlp_traj = TrajEncoder(traj_feat_dim=traj_feat_dim, num_steps=num_steps, wpt_dim=wpt_dim)
        self.mlp_cp = nn.Linear(3, cp_feat_dim) # contact point

        # Not in use
        # self.dir_encoder = nn.Linear(6, dir_feat_dim) # TODO: may be used in the future

        self.all_encoder = AllEncoder(
                                pcd_feat_dim=pcd_feat_dim, traj_feat_dim=traj_feat_dim, cp_feat_dim=cp_feat_dim,
                                hidden_dim=hidden_dim, z_feat_dim=z_feat_dim
                            ) # CVAE encoder
        self.all_decoder = AllDecoder(
                                pcd_feat_dim=pcd_feat_dim, cp_feat_dim=cp_feat_dim,
                                z_feat_dim=z_feat_dim, hidden_dim=hidden_dim, 
                                num_steps=num_steps, wpt_dim=wpt_dim
                            ) # CVAE decoder
        self.MSELoss = nn.MSELoss(reduction='mean')

        self.num_steps = num_steps
        self.wpt_dim = wpt_dim

        self.lbd_kl = lbd_kl
        self.lbd_recon = lbd_recon
        self.lbd_dir = lbd_dir
        self.kl_annealing = kl_annealing

        self.dataset_type = dataset_type # 0 for absolute, 1 for residule

    # pcs: B x N x 3 (float), with the 0th point to be the query point
    # pred_result_logits: B, pcs_feat: B x F x N
    def forward(self, pcs, traj, contact_point):
        pcs = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs)

        f_s = whole_feats[:, :, 0]
        f_cp = self.mlp_cp(contact_point)
        f_traj = self.mlp_traj(traj)

        z_all, mu, logvar = self.all_encoder(f_s, f_traj, f_cp)
        recon_traj = self.all_decoder(f_s, f_cp, z_all)

        return recon_traj, mu, logvar

    def sample(self, pcs, contact_point):
        batch_size = pcs.shape[0]
        f_cp = self.mlp_cp(contact_point)
        z_all = torch.Tensor(torch.randn(batch_size, self.z_dim)).cuda()

        pcs = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs)
        f_s = whole_feats[:, :, 0]

        recon_traj = self.all_decoder(f_s, f_cp, z_all)
        ret_traj = torch.zeros(recon_traj.shape)
        ret_traj = recon_traj
        ret_traj[:, 0] = contact_point

        return ret_traj

    def get_loss(self, pcs, traj, contact_point, lbd_kl=1.0):
        batch_size = traj.shape[0]
        recon_traj, mu, logvar = self.forward(pcs, traj, contact_point)

        recon_loss = torch.Tensor([0])
        dir_loss = torch.Tensor([0])
        nn_loss = torch.Tensor([0])
        if self.dataset_type == 0: # absolute 
            recon_wps = recon_traj
            input_wps = traj
            # recon_loss = self.MSELoss(recon_wps.view(batch_size, self.num_steps * self.wpt_dim), input_wps.view(batch_size, self.num_steps * self.wpt_dim))

            bound1 = int(0.25 * self.num_steps)
            bound2 = int(0.5 * self.num_steps)
            recon_loss_0 = self.MSELoss(recon_wps[:, :bound1].view(batch_size, bound1 * self.wpt_dim), input_wps[:, :bound1].view(batch_size, bound1 * self.wpt_dim))
            recon_loss_1 = self.MSELoss(recon_wps[:, bound1:bound2].view(batch_size, (bound2 - bound1) * self.wpt_dim), input_wps[:, bound1:bound2].view(batch_size, (bound2 - bound1) * self.wpt_dim))
            recon_loss_2 = self.MSELoss(recon_wps[:, bound2:].view(batch_size, (self.num_steps - bound2) *  self.wpt_dim), input_wps[:, bound2:].view(batch_size, (self.num_steps - bound2) * self.wpt_dim))
            recon_loss = recon_loss_0 + 0.5 * recon_loss_1 + 0.25 * recon_loss_2

        if self.dataset_type == 1: # residualrecon_dir = recon_traj[:, 0, :]
            input_first_wpt = traj[:, 0]
            recon_first_wpt = recon_traj[:, 0]
            first_loss = self.MSELoss(input_first_wpt.view(batch_size, self.wpt_dim), recon_first_wpt.view(batch_size, self.wpt_dim))

            input_wps = traj[:, 1:]
            recon_wps = recon_traj[:, 1:]

            # recon loss
            bound1 = int(0.25 * self.num_steps)
            bound2 = int(0.5 * self.num_steps)
            recon_loss_0 = self.MSELoss(recon_wps[:, :bound1].view(batch_size, bound1 * self.wpt_dim), input_wps[:, :bound1].view(batch_size, bound1 * self.wpt_dim))
            recon_loss_1 = self.MSELoss(recon_wps[:, bound1:bound2].view(batch_size, (bound2 - bound1) * self.wpt_dim), input_wps[:, bound1:bound2].view(batch_size, (bound2 - bound1) * self.wpt_dim))
            recon_loss_2 = self.MSELoss(recon_wps[:, bound2:].view(batch_size, (self.num_steps - bound2) *  self.wpt_dim), input_wps[:, bound2:].view(batch_size, (self.num_steps - bound2) * self.wpt_dim))
            wpt_loss = recon_loss_0 + 0.5 * recon_loss_1 + 0.25 * recon_loss_2
            recon_loss = 100 * first_loss + wpt_loss

        kl_loss = KL(mu, logvar)
        losses = {}
        losses['dir'] = dir_loss
        losses['kl'] = kl_loss
        losses['recon'] = recon_loss
        losses['nn'] = nn_loss

        if self.kl_annealing == 0:
            losses['total'] = kl_loss * self.lbd_kl + recon_loss * self.lbd_recon
        elif self.kl_annealing == 1:
            losses['total'] = kl_loss * lbd_kl + recon_loss * self.lbd_recon

        return losses
    
