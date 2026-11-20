import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
from utils import *
import torch
import torch.nn as nn
import warnings
import logging
import matplotlib.pyplot as plt
from net import MeshDeformation, SpatialTransformer_CA, MMFusion

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

phase1_ckpt_path = r"./model/model_phase1.pth"
phase2_ckpt_path = r"./model/model_phase2.pth"

test_IR_folder = 'IR_warp'
test_RGB_folder = 'RGB'
save_folder_name = 'LLVIP'

def save_registration_field(image_A, registration_field, output_dir, idx, step=5):
    plt.ioff()

    H, W, _ = registration_field.shape

    X, Y = np.meshgrid(np.arange(0, W, step), np.arange(0, H, step))

    U = registration_field[::step, ::step, 0]
    V = registration_field[::step, ::step, 1]

    X_new = X + U
    Y_new = Y + V

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)
    ax.set_xlim([0, W])
    ax.set_ylim([H, 0])
    ax.set_title('Deformed Grid')
    plt.savefig(os.path.join(output_dir, f'deformed_grid_{idx}'))
    plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(15, 10))

    axs[0].imshow(image_A, cmap='gray')
    axs[0].plot(X, Y, color='r', linewidth=0.5)
    axs[0].plot(X.T, Y.T, color='r', linewidth=0.5)
    axs[0].set_title('Original Image with Grid')

    axs[1].imshow(image_A, cmap='gray')
    axs[1].plot(X_new, Y_new, color='b', linewidth=0.5)
    axs[1].plot(X_new.T, Y_new.T, color='b', linewidth=0.5)
    axs[1].set_title('Deformed Image with Grid')
    axs[1].set_xlim([0, W])
    axs[1].set_ylim([H, 0])

    plt.savefig(os.path.join(output_dir, f'deformed_image_with_grid_{idx}'))
    plt.close('all')


for dataset_name in ['LLVIP']:
    print("\n" * 2 + "=" * 80)
    model_name1 = "ours_align    "
    print("The test result of " + save_folder_name  + " for " + dataset_name + ' :')
    
    test_folder = os.path.join('./dataset', dataset_name, 'test')
    test_out_folder = os.path.join('./test_result_align', dataset_name + save_folder_name)
    test_out_folder_rgb = os.path.join('./test_result_align', dataset_name + save_folder_name)
    aligned_ir_out_folder = os.path.join('./test_result_align', dataset_name + save_folder_name)
    aligned_rgb_out_folder = os.path.join('./test_result_align', dataset_name + save_folder_name)
    flow_out_folder = os.path.join('./test_result_align', dataset_name + save_folder_name)
    aligned_rgb_3ch_out_folder = os.path.join('./test_result_align', dataset_name + save_folder_name)

    if not os.path.exists(flow_out_folder):
        os.makedirs(flow_out_folder)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Using device: ", device)

    flow_t2v = nn.DataParallel(MeshDeformation()).to(device)
    flow_v2t = nn.DataParallel(MeshDeformation()).to(device)
    flow_model = nn.DataParallel(MeshDeformation()).to(device)
    G_T2V = nn.DataParallel(SpatialTransformer_CA()).to(device)
    G_V2T = nn.DataParallel(SpatialTransformer_CA()).to(device)
    FusionNet = nn.DataParallel(MMFusion()).to(device)

    if phase2_ckpt_path is not None:
        print("phase2_ckpt_path is not None, loading phase2 checkpoint")
        flow_t2v.load_state_dict(torch.load(phase1_ckpt_path)['flow_t2v'])
        flow_v2t.load_state_dict(torch.load(phase1_ckpt_path)['flow_v2t'])
        G_T2V.load_state_dict(torch.load(phase1_ckpt_path)['G_T2V'])
        G_V2T.load_state_dict(torch.load(phase1_ckpt_path)['G_V2T'])
        FusionNet.load_state_dict(torch.load(phase2_ckpt_path)['fusion_model'])
    else:
        print("phase2_ckpt_path is None, loading phase1 checkpoint")
        flow_t2v.load_state_dict(torch.load(phase1_ckpt_path)['flow_t2v'])
        flow_v2t.load_state_dict(torch.load(phase1_ckpt_path)['flow_v2t'])
        G_T2V.load_state_dict(torch.load(phase1_ckpt_path)['G_T2V'])
        G_V2T.load_state_dict(torch.load(phase1_ckpt_path)['G_V2T'])
        FusionNet.load_state_dict(torch.load(phase1_ckpt_path)['fusion_model'])
    flow_t2v.eval()
    flow_v2t.eval()
    flow_model.eval()
    G_T2V.eval()
    G_V2T.eval()
    FusionNet.eval()

    with torch.no_grad():
        for img_name in os.listdir(os.path.join(test_folder, test_IR_folder)):
            data_ir = image_read_cv2(os.path.join(test_folder, test_IR_folder, img_name), mode='GRAY')[
                          np.newaxis, np.newaxis, ...] / 255.0
            data_rgb = image_read_cv2(os.path.join(test_folder, test_RGB_folder, img_name), mode='YCrCb')
            data_rgb = data_rgb[np.newaxis, np.newaxis, ...] / 255.0
            data_ir, data_rgb = torch.FloatTensor(data_ir).to(device), torch.FloatTensor(data_rgb).to(device)
            
            data_VIS_BGR = cv2.imread(os.path.join(test_folder, test_RGB_folder, img_name))
            _, data_VIS_Cr, data_VIS_Cb = cv2.split(cv2.cvtColor(data_VIS_BGR, cv2.COLOR_BGR2YCrCb))

            flow, neg_flow = flow_t2v(data_rgb, data_ir)
            flow_vt, neg_flow_vt = flow_v2t(data_ir, data_rgb)
            ir_aligned = G_V2T(data_ir, flow)
            rgb_aligned = G_T2V(data_rgb, flow_vt)

            flow = flow.squeeze(0).permute(1, 2, 0).cpu().numpy()
            image_A = data_rgb.squeeze().cpu().numpy()
            flow = save_registration_field(image_A, flow, flow_out_folder, img_name)

            save_rgb_aligned = np.squeeze(rgb_aligned * 255).cpu().numpy().astype(np.uint8)
            save_ir_aligned = np.squeeze(ir_aligned * 255).cpu().numpy().astype(np.uint8)
            img_save(save_rgb_aligned, img_name.split(sep='.')[0], aligned_rgb_out_folder)
            img_save(save_ir_aligned, img_name.split(sep='.')[0], aligned_ir_out_folder)

            if phase2_ckpt_path is not None:
                Fuse, rgb_base, rgb_detail, ir_base, ir_detail, Fuse_feature, rgb_feature, ir_feature = FusionNet(rgb=rgb_aligned, ir=data_ir, train_phase1=False)
                data_fuse = (Fuse - torch.min(Fuse)) / (torch.max(Fuse) - torch.min(Fuse))
                fi = np.squeeze(data_fuse * 255).cpu().numpy().astype(np.uint8)
                img_save(fi, img_name.split(sep='.')[0], test_out_folder)
                ycrcb_fi = np.dstack((fi, data_VIS_Cr, data_VIS_Cb))
                rgb_fi = cv2.cvtColor(ycrcb_fi, cv2.COLOR_YCrCb2RGB)
                img_save(rgb_fi, img_name.split(sep='.')[0], test_out_folder_rgb)
