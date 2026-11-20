import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import kornia.filters as filters
import scipy.spatial
# import faiss ()

class MakeDataset(Dataset):
    def __init__(self, rgb_dir, ir_dir, is_train=False):
        self.rgb_dir = rgb_dir
        self.ir_dir = ir_dir
        self.train = is_train
        self.file_names = os.listdir(rgb_dir)

        if self.train:
            self.rgb_transform = transforms.Compose([
                transforms.Resize((360, 320)),
            ])
            self.ir_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((360, 320)),
            ])
        else:
            self.rgb_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            self.ir_transform = transforms.Compose([
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        rgb_file = os.path.join(self.rgb_dir, self.file_names[idx])
        ir_file = os.path.join(self.ir_dir, self.file_names[idx])
        rgb_image = cv2.imread(rgb_file, 1)  # BGR
        (B, G, R) = cv2.split(rgb_image)
        Y = (0.299 * R + 0.587 * G + 0.114 * B) / 255.0
        Cr = (R - Y) * 0.713 + 0.5
        Cb = (B - Y) * 0.564 + 0.5
        Y_tensor = torch.tensor(Y, dtype=torch.float32, device='cuda')
        Y_tensor = Y_tensor.unsqueeze(0)
        Cr_tensor = torch.tensor(Cr, dtype=torch.float32, device='cuda')
        Cb_tensor = torch.tensor(Cb, dtype=torch.float32, device='cuda')
        CrCb_tensor = torch.cat([Cr_tensor.unsqueeze(0), Cb_tensor.unsqueeze(0)], dim=0)

        ir_image = cv2.imread(ir_file, 0)  

        ycrcb_image = self.rgb_transform(Y_tensor)
        ir_image = self.ir_transform(ir_image)

        return ycrcb_image, CrCb_tensor, ir_image

def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, H, W = img1.shape
    #N2, C2, H2, W2 = img2.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1 **
                                                                      2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()

def NCC(img1, img2):
    H, W = img1.shape[2:]
    mean1 = img1.mean(dim=(2, 3), keepdim=True)
    mean2 = img2.mean(dim=(2, 3), keepdim=True)
    mean1_repeated = mean1.repeat(1, 1, img1.shape[2], img1.shape[3])
    mean2_repeated = mean2.repeat(1, 1, img2.shape[2], img2.shape[3])

    diff1 = img1 - mean1_repeated
    diff2 = img2 - mean2_repeated
    numerator = torch.mul(diff1, diff2)

    mean1 = torch.mean(img1, dim=(2, 3), keepdim=True)
    mean2 = torch.mean(img2, dim=(2, 3), keepdim=True)

    std1 = torch.mean(torch.sqrt((img1 - mean1) ** 2),dim=(2, 3), keepdim=True)
    std2 = torch.mean(torch.sqrt((img2 - mean2) ** 2),dim=(2, 3), keepdim=True)
    std1_repeated = std1.repeat(1, 1, img1.shape[2], img1.shape[3])
    std2_repeated = std2.repeat(1, 1, img2.shape[2], img2.shape[3])
    denominator = torch.mul(std1_repeated, std2_repeated)

    out = torch.mean(torch.div(numerator, denominator + 1e-7))
    return out

class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self,image_vis,image_ir,generate_img):
        image_y=image_vis[:,:1,:,:]
        x_in_max=torch.max(image_y,image_ir)
        loss_in=F.l1_loss(x_in_max,generate_img)
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        loss_total=loss_in+10*loss_grad
        return loss_total,loss_in,loss_grad

class Sobelxyloss(nn.Module):
    def __init__(self):
        super(Sobelxyloss, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self,image_vis,image_ir):
        image_y=image_vis[:,:1,:,:]
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        loss=F.l1_loss(y_grad,ir_grad)
        return loss

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]]
        kernely = [[-1, -2, -1],
                  [0,0 , 0],
                  [1, 2, 1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False) .cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)

        return sobelx+sobely

def batch_rgb2y(img):
    y = img[:, 0:1, :, :] * 0.299 + img[:, 1:2, :, :] * 0.587 + img[:, 2:3, :, :] * 0.114
    return y
    
class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def loss(self, _, y_pred):
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()

def smooth_loss(flow):
    dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
    dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
    dy = dy * dy
    dx = dx * dx
    return (torch.mean(dx) + torch.mean(dy)) / 2.0

def image_read_cv2(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    #img_BGR =cv2.resize(img_BGR, (512, 320))
    assert mode == 'BGR' or mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
        img = np.transpose(img,(2,0,1))
        return img
    elif mode == 'GRAY':
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
        return img
    elif mode == 'BGR':
        return img_BGR
    elif mode == 'YCrCb':
        #img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
        (B, G, R) = cv2.split(img_BGR)
        Y = np.round(0.299 * R + 0.587 * G + 0.114 * B)
        #Cr = (R - Y) * 0.713 + 0.5
        #Cb = (B - Y) * 0.564 + 0.5
        #CrCb = np.round(cv2.merge([Cr,Cb]))
        #CrCb = CrCb.astype(np.uint8)
        return Y#, CrCb

def img_save(image,imagename,savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    # Gray_pic
    imsave(os.path.join(savepath, "{}.png".format(imagename)),image)