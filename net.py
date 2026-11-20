import torch
import torch.nn as nn
import numpy as np
import cv2
import torch.nn.functional as F
from utils import *
import torch.nn.functional as nnf
from torch.distributions.normal import Normal
from torch.autograd import Variable
from einops import rearrange
import numbers
from FlowNet_model_level import FlowNet

class MMFusion(nn.Module):
    def __init__(self):
        super(MMFusion, self).__init__()

        self.encoder = Restormer_Encoder()
        self.decoder = Restormer_Decoder()
        self.base_feature = BaseFeatureExtraction(dim=64, num_heads=8)
        self.detile_feature = DetailFeatureExtraction(num_layers=1)

    def forward(self, rgb, ir, train_phase1=True):
        if train_phase1:
            rgb_base, rgb_detail, _ = self.encoder(rgb)
            ir_base, ir_detail, _ = self.encoder(ir)

            rgb_pre, _ = self.decoder(rgb, rgb_base, rgb_detail)
            ir_pre, _ = self.decoder(ir, ir_base, ir_detail)

            return rgb_base, rgb_detail, ir_base, ir_detail, rgb_pre, ir_pre
        else:
            rgb_base, rgb_detail, rgb_feature = self.encoder(rgb)
            ir_base, ir_detail, ir_feature = self.encoder(ir)
            base_feature = self.base_feature(rgb_base + ir_base)
            detail_feature = self.detile_feature(rgb_detail + ir_detail)
            Fuse, Fuse_feature = self.decoder(rgb, base_feature, detail_feature)

            return Fuse, rgb_base, rgb_detail, ir_base, ir_detail, Fuse_feature, rgb_feature, ir_feature

class MeshDeformation(nn.Module):
    def __init__(self):
        super(MeshDeformation, self).__init__()
        self.unet = UNet()
        self.flowenet = FlowNet(in_ch=2)
        self.flow = nn.Conv2d(16, 2, kernel_size=3, padding=1)
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        self.vt_generator = nn.Conv2d(16, 2, kernel_size=3, padding=1)
        self.vt_generator.weight = nn.Parameter(Normal(0, 1e-5).sample(self.vt_generator.weight.shape))
        self.vt_generator.bias = nn.Parameter(torch.zeros(self.vt_generator.bias.shape))

    def forward(self, rgb, ir):
        x_input = torch.cat((rgb, ir), dim=1)
        x = self.unet(x_input)

        flow_field = self.flow(x)

        pos_flow = flow_field

        neg_flow = -pos_flow

        return pos_flow, neg_flow

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv_input = nn.Conv2d(2, 16, 3, padding=1)
        self.conv_block1 = ConvBlock(16, 32)
        self.conv_block2 = ConvBlock(32, 32)
        self.conv_block3 = ConvBlock(32, 64)
        self.conv_block4 = ConvBlock(64, 128)
        self.conv_block5 = ConvBlock(128, 128)
        self.conv_block6 = ConvBlock(128, 64)
        self.conv_block7 = ConvBlock(64, 32)
        self.conv_block8 = ConvBlock(256, 128)
        self.conv_block9 = ConvBlock(128, 256)
        self.conv_block10 = ConvBlock(256, 512)
        self.conv_block11 = ConvBlock(512, 256)

        self.up_trans1 = self.up_conv(512, 256)
        self.up_trans2 = self.up_conv(256, 128)
        self.up_trans3 = self.up_conv(128, 64)
        self.up_trans4 = self.up_conv(64, 32)
        self.up_trans5 = self.up_conv(32, 16)

        self.up_conv1 = self.double_conv(512, 256)
        self.up_conv2 = self.double_conv(256, 128)
        self.up_conv3 = self.double_conv(128, 64)
        self.up_conv4 = self.double_conv(64, 32)
        self.up_conv5 = self.double_conv(32, 16)

        self.conv_block12 = ConvBlock(32, 16)
        self.conv_block13 = ConvBlock(16, 16)
        self.conv_block14 = ConvBlock(16, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

    def forward(self, input):
        x_1 = self.conv_input(input)

        x_2 = self.conv_block1(x_1)
        x_2 = self.maxpool(x_2)

        x_3 = self.conv_block3(x_2)
        x_3 = self.maxpool(x_3)

        x_4 = self.conv_block4(x_3)
        x_4 = self.maxpool(x_4)

        x_5 = self.conv_block9(x_4)
        x_5 = self.maxpool(x_5)

        x_6 = self.conv_block10(x_5)

        x_7 = self.up_trans1(x_6)
        x_7 = nn.Upsample(size=(x_5.shape[2], x_5.shape[3]), mode='nearest')(x_7)
        x_7 = torch.cat((x_7, x_5),dim=1)
        x_7 = self.up_conv1(x_7)

        x_8 = self.up_trans2(x_7)
        x_8 = nn.Upsample(size=(x_4.shape[2], x_4.shape[3]), mode='nearest')(x_8)
        x_8 = torch.cat((x_8, x_4),dim=1)
        x_8 = self.up_conv2(x_8)

        x_9 = self.up_trans3(x_8)
        x_9 = nn.Upsample(size=(x_3.shape[2], x_3.shape[3]), mode='nearest')(x_9)
        x_9 = torch.cat((x_9, x_3),dim=1)
        x_9 = self.up_conv3(x_9)

        x_10 = self.up_trans4(x_9)
        x_10 = nn.Upsample(size=(x_2.shape[2], x_2.shape[3]), mode='nearest')(x_10)
        x_10 = torch.cat((x_10, x_2),dim=1)
        x_10 = self.up_conv4(x_10)

        x_11 = self.up_trans5(x_10)
        x_11 = nn.Upsample(size=(x_1.shape[2], x_1.shape[3]), mode='nearest')(x_11)
        x_11 = torch.cat((x_11, x_1),dim=1)
        x_11 = self.up_conv5(x_11)

        x_out = self.conv_block13(x_11)
        x_out = self.conv_block13(x_out)

        return x_out

    def up_conv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

class SpatialTransformer_CA(nn.Module):
    """
    N-D Spatial Transformer (modified for non-fixed size) with learnable parameters.
    """

    def __init__(self, in_channels=1, out_channels=2, kernel_size=3, mode='bilinear'):
        super(SpatialTransformer_CA, self).__init__()
        self.mode = mode

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_size, padding=1)
        self.conv3 = nn.Conv2d(32, out_channels, kernel_size=kernel_size, padding=1)

    def forward(self, src, flow=None):
        if flow is None:
            flow = self.conv1(src)
            flow = nnf.relu(flow)
            flow = self.conv2(flow)
            flow = nnf.relu(flow)
            flow = self.conv3(flow)

        src_shape = src.shape[2:]

        vectors = [torch.arange(0, dim) for dim in src_shape]
        grid = torch.meshgrid(vectors)
        grid = torch.stack(grid)
        grid = grid.type(torch.FloatTensor)
        grid = torch.unsqueeze(grid, 0) .cuda()

        new_locs = grid + flow

        for i in range(len(src_shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (src_shape[i] - 1) - 0.5)

        if len(src_shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(src_shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=False, mode=self.mode)

class Warper2d(nn.Module):
    def __init__(self):
        super(Warper2d, self).__init__()

        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        img_smp: [B, 1, H2, W2] (image for sampling, size 44)
        flow: [B, 2, H1, W1] flow predicted from source image pair
        """

    def forward(self, img, flow):
        H, W = flow.size()[2], flow.size()[3]

        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, H, W)
        yy = yy.view(1, H, W)
        grid = torch.cat((xx, yy), 0).float()

        device = img.device

        if img.is_cuda:
            grid = grid.to(device)

        vgrid = Variable(grid, requires_grad=False) + flow
        vgrid[:, 0] = 2.0 * vgrid[:, 0] / (H - 1) - 1.0
        vgrid[:, 1] = 2.0 * vgrid[:, 1] / (W - 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = F.grid_sample(img, vgrid)

        return output

class DeformationDiscriminator(nn.Module):
    def __init__(self, in_channels=2, n_layers=3, out_channels=64):
        super(DeformationDiscriminator, self).__init__()

        self.in_channels = in_channels
        sequence = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(out_channels * nf_mult_prev, out_channels * nf_mult, kernel_size=4, stride=2, padding=1, bias=True),
                nn.InstanceNorm2d(out_channels * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(out_channels * nf_mult_prev, out_channels * nf_mult, kernel_size=4, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(out_channels * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(out_channels * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*sequence)
    def forward(self, input):
        """Standard forward."""
        return self.model(input)
class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.main = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        out = self.bn(out)
        return out

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob +\
                    torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class AttentionBase(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False, ):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.proj(out)
        return out

class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 ffn_expansion_factor=2,
                 bias=False):
        super().__init__()
        hidden_features = int(in_features * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            in_features, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, in_features, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class BaseFeatureExtraction(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor=1.,
                 qkv_bias=False, ):
        super(BaseFeatureExtraction, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias, )
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = Mlp(in_features=dim,
                       ffn_expansion_factor=ffn_expansion_factor, )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            nn.ReLU6(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
        )

    def forward(self, x):
        return self.bottleneckBlock(x)

class DetailNode(nn.Module):
    def __init__(self):
        super(DetailNode, self).__init__()
        self.theta_phi = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.shffleconv = nn.Conv2d(64, 64, kernel_size=1,
                                    stride=1, padding=0, bias=True)

    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:x.shape[1]]
        return z1, z2

    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(
            self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2

class DetailFeatureExtraction(nn.Module):
    def __init__(self, num_layers=3):
        super(DetailFeatureExtraction, self).__init__()
        INNmodules = [DetailNode() for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)

    def forward(self, x):
        z1, z2 = x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:x.shape[1]]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)

import numbers

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):

        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x

class Restormer_Encoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):
        super(Restormer_Encoder, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.baseFeature = BaseFeatureExtraction(dim=dim, num_heads=heads[2])
        self.detailFeature = DetailFeatureExtraction()

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        base_feature = self.baseFeature(out_enc_level1)
        detail_feature = self.detailFeature(out_enc_level1)
        return base_feature, detail_feature, out_enc_level1

class Restormer_Decoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(Restormer_Decoder, self).__init__()
        self.reduce_channel = nn.Conv2d(int(dim*2), int(dim), kernel_size=1, bias=bias)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.output = nn.Sequential(
            nn.Conv2d(int(dim), int(dim)//2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim)//2, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias),)
        self.sigmoid = nn.Sigmoid()
    def forward(self, inp_img, base_feature, detail_feature):
        out_enc_level0 = torch.cat((base_feature, detail_feature), dim=1)
        out_enc_level0 = self.reduce_channel(out_enc_level0)
        out_enc_level1 = self.encoder_level2(out_enc_level0)
        if inp_img is not None:
            out_enc_level1 = self.output(out_enc_level1) + inp_img
        else:
            out_enc_level1 = self.output(out_enc_level1)
        return self.sigmoid(out_enc_level1), out_enc_level0

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=0),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=0),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)

class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num):
        super(GeneratorResNet, self).__init__()
        channels = input_shape[0]

        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, kernel_size=7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        for _ in range(num):
            model += [ResidualBlock(out_features)]

        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, kernel_size=7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator_cycleGAN(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator_cycleGAN, self).__init__()

        channels, height, width = input_shape

        patch_h, patch_w = height // 2 ** 3, width // 2 ** 3
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)

class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss
