import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss
from torch.utils.data import DataLoader
import random
import time
import datetime

from utils import *
from net import MeshDeformation, SpatialTransformer_CA, DeformationDiscriminator, MMFusion, GANLoss , Warper2d
from hyptorch.nn import ImageToHyperbolic,ImageFromHyperbolic,HyperbolicDistanceLayer
import kornia
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from loss_pfc import HyperbolicFeatureAlignmentLoss
from loss_sobel_hyper import CharbonnierHyperbolicLoss
from loss_hyper_global import GlobalHyperbolicLoss


num_epochs = 240
G_learning_rate = 0.0005
D_learning_rate = 0.00005
F_learning_rate = 0.0001
batch_size = 1
hpy_c=0.001
use_hpy = True

weight_decay = 0
phase_gap = 60

rgb_dir = './dataset/LLVIP/train/RGB'
ir_dir = './dataset/LLVIP/train/IR_warp'

dataset_name = rgb_dir.replace('./dataset/', '')

dataset_name = dataset_name.split('/', 1)[0]

save_path = './model/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter(log_dir='./logs')

dataset = MakeDataset(rgb_dir, ir_dir, is_train=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

flow_t2v = nn.DataParallel(MeshDeformation()).to(device)
flow_v2t = nn.DataParallel(MeshDeformation()).to(device)
G_T2V = nn.DataParallel(SpatialTransformer_CA()).to(device)
G_V2T = nn.DataParallel(SpatialTransformer_CA()).to(device)
D_F = nn.DataParallel(DeformationDiscriminator(in_channels=2, n_layers=3)).to(device)
D_V2T = nn.DataParallel(DeformationDiscriminator(in_channels=1, n_layers=3)).to(device)
D_T2V = nn.DataParallel(DeformationDiscriminator(in_channels=1, n_layers=3)).to(device)
fusion_model = nn.DataParallel(MMFusion()).to(device)

MSELoss = nn.MSELoss()
L1Loss = nn.L1Loss()
Loss_ssim = kornia.losses.SSIMLoss(11, reduction='mean')
criteria_fusion = Fusionloss()
criteria_sobelxy = Sobelxyloss()

Grad_loss = Grad().loss

D_criterion = GANLoss(gan_mode='lsgan').to('cuda')

hyperbolic_loss = HyperbolicFeatureAlignmentLoss(
    block_size=64,
    num_blocks=25,
    feature_dim=256,
    c=hpy_c
).to(device)

hyperbolic_edge_loss = CharbonnierHyperbolicLoss(c=hpy_c).to(device)
hyper_global_loss = GlobalHyperbolicLoss(feature_dim=256,c=hpy_c).to(device)

optimizer_flow_t2v = optim.AdamW(flow_t2v.parameters(), lr=G_learning_rate, weight_decay=weight_decay)
optimizer_flow_v2t = optim.AdamW(flow_v2t.parameters(), lr=G_learning_rate, weight_decay=weight_decay)
optimizer_G_T2V = optim.AdamW(G_T2V.parameters(), lr=G_learning_rate, weight_decay=weight_decay)
optimizer_G_V2T = optim.AdamW(G_V2T.parameters(), lr=G_learning_rate, weight_decay=weight_decay)

optimizer_D_T2V = optim.AdamW(D_T2V.parameters(), lr=D_learning_rate, weight_decay=weight_decay)
optimizer_D_V2T = optim.AdamW(D_V2T.parameters(), lr=D_learning_rate, weight_decay=weight_decay)
optimizer_fusion = optim.AdamW(fusion_model.parameters(), lr=F_learning_rate, weight_decay=0)

scheduler_lr_flow_t2v = optim.lr_scheduler.StepLR(optimizer_flow_t2v, step_size=20, gamma=0.5)
scheduler_lr_flow_v2t = optim.lr_scheduler.StepLR(optimizer_flow_v2t, step_size=20, gamma=0.5)
scheduler_lr_G_T2V = optim.lr_scheduler.StepLR(optimizer_G_T2V, step_size=20, gamma=0.5)
scheduler_lr_G_V2T = optim.lr_scheduler.StepLR(optimizer_G_V2T, step_size=20, gamma=0.5)

scheduler_lr_D_T2V = optim.lr_scheduler.StepLR(optimizer_D_T2V, step_size=20, gamma=0.5)
scheduler_lr_D_V2T = optim.lr_scheduler.StepLR(optimizer_D_V2T, step_size=20, gamma=0.5)
scheduler_lr_fusion = optim.lr_scheduler.StepLR(optimizer_fusion, step_size=10, gamma=0.5)

min_loss1 = float('inf')
min_loss1_2 =float('inf')
min_loss2 = float('inf')
checkpoint = {
    'flow_t2v': flow_t2v.state_dict(),
    'flow_v2t': flow_v2t.state_dict(),
    'G_T2V': G_T2V.state_dict(),
    'G_V2T': G_V2T.state_dict(),
    'D_F': D_F.state_dict(),
    'D_T2V': D_T2V.state_dict(),
    'D_V2T': D_V2T.state_dict(),
    'fusion_model': fusion_model.state_dict(),
}

print ('start training')
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    running_loss = 0.0

    for i, (rgb_batch, CrCb, ir_batch) in enumerate(dataloader):
        rgb_batch = rgb_batch.to(device)
        ir_batch = ir_batch.to(device)

        batch_size = rgb_batch.size(0)

        if epoch < phase_gap:

            D_T2V.train()
            D_V2T.train()
            flow_t2v.eval()
            flow_v2t.eval()
            G_T2V.eval()
            G_V2T.eval()

            optimizer_D_T2V.zero_grad()
            optimizer_D_V2T.zero_grad()

            with torch.no_grad():
                t2v_pos_flow, t2v_neg_flow = flow_t2v(rgb_batch, ir_batch)
                v2t_pos_flow, v2t_neg_flow = flow_v2t(ir_batch, rgb_batch)
                deform_ir1 = G_T2V(ir_batch, t2v_pos_flow)
                fake_ir2 = G_V2T(deform_ir1, v2t_pos_flow)
                deform_rgb1 = G_V2T(rgb_batch, t2v_neg_flow)
                fake_rgb2 = G_T2V(deform_rgb1, v2t_neg_flow)

            T2V_real_xy = kornia.filters.SpatialGradient()(rgb_batch)
            T2V_fake_xy = kornia.filters.SpatialGradient()(deform_ir1)
            V2T_real_xy = kornia.filters.SpatialGradient()(ir_batch)
            V2T_fake_xy = kornia.filters.SpatialGradient()(deform_rgb1)

            T2V_real_x = T2V_real_xy[:, :, 0, :, :]
            T2V_real_y = T2V_real_xy[:, :, 1, :, :]
            T2V_fake_x = T2V_fake_xy[:, :, 0, :, :]
            T2V_fake_y = T2V_fake_xy[:, :, 1, :, :]
            V2T_real_x = V2T_real_xy[:, :, 0, :, :]
            V2T_real_y = V2T_real_xy[:, :, 1, :, :]
            V2T_fake_x = V2T_fake_xy[:, :, 0, :, :]
            V2T_fake_y = V2T_fake_xy[:, :, 1, :, :]

            D_T2V_real_x = D_T2V(T2V_real_x)
            D_T2V_real_y = D_T2V(T2V_real_y)
            D_V2T_real_x = D_V2T(V2T_real_x)
            D_V2T_real_y = D_V2T(V2T_real_y)

            D_T2V_fake_x = D_T2V(T2V_fake_x.detach())
            D_T2V_fake_y = D_T2V(T2V_fake_y.detach())
            D_V2T_fake_x = D_V2T(V2T_fake_x.detach())
            D_V2T_fake_y = D_V2T(V2T_fake_y.detach())

            loss_d_r_t2v_x = D_criterion(D_T2V_real_x, True)
            loss_d_r_t2v_y = D_criterion(D_T2V_real_y, True)
            loss_d_r_v2t_x = D_criterion(D_V2T_real_x, True)
            loss_d_r_v2t_y = D_criterion(D_V2T_real_y, True)

            loss_d_f_t2v_x = D_criterion(D_T2V_fake_x, False)
            loss_d_f_t2v_y = D_criterion(D_T2V_fake_y, False)
            loss_d_f_v2t_x = D_criterion(D_V2T_fake_x, False)
            loss_d_f_v2t_y = D_criterion(D_V2T_fake_y, False)

            loss_d_r = loss_d_r_t2v_x + loss_d_r_t2v_y + loss_d_r_v2t_x + loss_d_r_v2t_y
            loss_d_f = loss_d_f_t2v_x + loss_d_f_t2v_y + loss_d_f_v2t_x + loss_d_f_v2t_y
            loss_d = (loss_d_r + loss_d_f) * 0.5

            loss_d.backward()
            optimizer_D_T2V.step()
            optimizer_D_V2T.step()

            flow_t2v.train()
            flow_v2t.train()
            G_T2V.train()
            G_V2T.train()
            D_T2V.eval()
            D_V2T.eval()

            optimizer_flow_t2v.zero_grad()
            optimizer_flow_v2t.zero_grad()
            optimizer_G_T2V.zero_grad()
            optimizer_G_V2T.zero_grad()

            t2v_pos_flow, t2v_neg_flow = flow_t2v(rgb_batch, ir_batch)
            v2t_pos_flow, v2t_neg_flow = flow_v2t(ir_batch, rgb_batch)

            deform_ir1 = G_T2V(ir_batch, t2v_pos_flow)
            fake_ir2 = G_V2T(deform_ir1, v2t_pos_flow)

            deform_rgb1 = G_V2T(rgb_batch, t2v_neg_flow)
            fake_rgb2 = G_T2V(deform_rgb1, v2t_neg_flow)

            T2V_fake_xy = kornia.filters.SpatialGradient()(deform_ir1)
            V2T_fake_xy = kornia.filters.SpatialGradient()(deform_rgb1)

            T2V_fake_x = T2V_fake_xy[:, :, 0, :, :]
            T2V_fake_y = T2V_fake_xy[:, :, 1, :, :]
            V2T_fake_x = V2T_fake_xy[:, :, 0, :, :]
            V2T_fake_y = V2T_fake_xy[:, :, 1, :, :]

            D_T2V_fake_x = D_T2V(T2V_fake_x)
            D_T2V_fake_y = D_T2V(T2V_fake_y)
            D_V2T_fake_x = D_V2T(V2T_fake_x)
            D_V2T_fake_y = D_V2T(V2T_fake_y)

            loss_g_adv_t2v_x = D_criterion(D_T2V_fake_x, True)
            loss_g_adv_t2v_y = D_criterion(D_T2V_fake_y, True)
            loss_g_adv_v2t_x = D_criterion(D_V2T_fake_x, True)
            loss_g_adv_v2t_y = D_criterion(D_V2T_fake_y, True)

            loss_g_adv = loss_g_adv_t2v_x + loss_g_adv_t2v_y + loss_g_adv_v2t_x + loss_g_adv_v2t_y

            cycle_consistency_ir = L1Loss(ir_batch, fake_ir2)
            cycle_consistency_rgb = L1Loss(rgb_batch, fake_rgb2)
            loss_cycle = cycle_consistency_ir + cycle_consistency_rgb

            loss_flow_smooth = smooth_loss(t2v_pos_flow) + smooth_loss(v2t_pos_flow)

            loss_sobelxy = hyperbolic_edge_loss(deform_ir1, rgb_batch, is_hpy=use_hpy) + hyperbolic_edge_loss(deform_rgb1, ir_batch, is_hpy=use_hpy)

            loss_hyper_global_loss = hyper_global_loss(rgb_batch, deform_ir1, is_hpy=use_hpy) + hyper_global_loss(ir_batch, deform_rgb1, is_hpy=use_hpy)

            loss_l1 = L1Loss(fake_ir2, deform_rgb1) + L1Loss(fake_rgb2, deform_ir1)

            loss = loss_g_adv + loss_cycle +  loss_sobelxy + loss_l1 + loss_flow_smooth*0.01 + loss_hyper_global_loss

            loss.backward()
            optimizer_flow_t2v.step()
            optimizer_flow_v2t.step()
            optimizer_G_T2V.step()
            optimizer_G_V2T.step()

            scheduler_lr_G_T2V.step()
            scheduler_lr_G_V2T.step()
            scheduler_lr_flow_t2v.step()
            scheduler_lr_flow_v2t.step()
            scheduler_lr_D_T2V.step()
            scheduler_lr_D_V2T.step()

            running_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}]'
                    f' D_Loss: {loss_d.item():.4f} (Real: {loss_d_r.item():.4f}, Fake: {loss_d_f.item():.4f})'
                    f' Loss: {loss.item():.4f}, Adv_Loss: {loss_g_adv.item():.4f}'
                    f' Cycle_Loss: {loss_cycle.item():.4f}, Flow_Loss: {loss_flow_smooth.item():.4f}'
                    f' Sobelxy_Loss: {loss_sobelxy.item():.4f}'

                    f' L1_Loss: {loss_l1.item():.4f}, Hyper_Global_Loss: {loss_hyper_global_loss.item():.4f}')

        else:
            fusion_model.train()
            flow_t2v.eval()
            optimizer_fusion.zero_grad()

            pos_flow, _ = flow_t2v(rgb_batch, ir_batch)
            ir_aligned = G_T2V(ir_batch, pos_flow)

            Fuse, rgb_base, rgb_detail, ir_base, ir_detail, Fuse_feature, rgb_feature, ir_feature = fusion_model(
                rgb=rgb_batch, ir=ir_aligned, train_phase1=False)

            cc_b = cc(rgb_base, ir_base)
            cc_d = cc(rgb_detail, ir_detail)
            decomp_loss = (cc_d) ** 2 / (1.01 + cc_b)
            mse_rgb = 5 * Loss_ssim(rgb_batch, Fuse) + L1Loss(rgb_batch, Fuse)
            mse_ir = 5 * Loss_ssim(ir_aligned, Fuse) + L1Loss(ir_aligned, Fuse)

            fusion_loss, _, _ = criteria_fusion(rgb_batch, ir_aligned, Fuse)

            loss = fusion_loss + 2 * decomp_loss + mse_rgb + mse_ir

            loss.backward()
            optimizer_fusion.step()

            running_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}]'
                    f' Loss: {loss.item():.4f}'
                    f' Fusion_Loss: {fusion_loss.item():.4f}'
                    f' Decomp_Loss: {decomp_loss.item():.4f}')

    if epoch < phase_gap:
        writer.add_scalar('Loss/loss_smooth', loss_flow_smooth, epoch)
        writer.add_scalar('Loss/loss_d', loss_d, epoch)
        writer.add_scalar('Loss/loss_d_r_t2v_x', loss_d_r_t2v_x, epoch)
        writer.add_scalar('Loss/loss_d_r_t2v_y', loss_d_r_t2v_y, epoch)
        writer.add_scalar('Loss/loss_d_r_v2t_x', loss_d_r_v2t_x, epoch)
        writer.add_scalar('Loss/loss_d_r_v2t_y', loss_d_r_v2t_y, epoch)
        writer.add_scalar('Loss/loss_d_f_t2v_x', loss_d_f_t2v_x, epoch)
        writer.add_scalar('Loss/loss_d_f_t2v_y', loss_d_f_t2v_y, epoch)
        writer.add_scalar('Loss/loss_d_f_v2t_x', loss_d_f_v2t_x, epoch)
        writer.add_scalar('Loss/loss_d_f_v2t_y', loss_d_f_v2t_y, epoch)
        writer.add_scalar('Loss/hyper_global_loss', loss_hyper_global_loss, epoch)
    else:
        writer.add_scalar('Loss/loss_fusion', fusion_loss, epoch)
        writer.add_scalar('Loss/decomp_loss', decomp_loss, epoch)

    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time
    epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
    print(f'Time: {epoch_time_str}')

    avg_loss = running_loss / len(dataloader)
    print(avg_loss)
    writer.add_scalar('Loss/avg_loss', avg_loss, epoch)
    if epoch < phase_gap :
        min_loss1 = avg_loss
        torch.save(checkpoint, os.path.join(save_path, 'model_phase1.pth'))
        print(f'Phase1: Saved min model with loss: {min_loss1:.4f}')

    if epoch >= phase_gap:
        if avg_loss < min_loss2 :
            min_loss2 = avg_loss
            torch.save(checkpoint, os.path.join(save_path, 'model_phase2.pth'))
            print(f'Phase2: Saved min model with loss: {min_loss2:.4f}')
writer.close()

print(f"Completion of {dataset_name} training")
