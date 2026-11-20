import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn as nn
import torchvision.models as models
import kornia
import numpy as np
from math import acosh
from hyptorch.nn import HyperbolicDistanceLayer
from hyptorch.nn import ToPoincare

class HyperbolicFeatureAlignmentLoss(nn.Module):
    def __init__(self, block_size=32, num_blocks=10, feature_dim=256, c=1.0):
        super(HyperbolicFeatureAlignmentLoss, self).__init__()
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.feature_dim = feature_dim

        vgg16 = models.vgg16(pretrained=False)
        vgg16.load_state_dict(torch.load('./hyptorch/vgg16-397923af.pth'))  # 加载本地权重

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        with torch.no_grad():
            self.conv1.weight.data = torch.sum(vgg16.features[0].weight.data, dim=1, keepdim=True)
        self.conv1.requires_grad_(False)

        self.feature_extractor = nn.Sequential(
            self.conv1,
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # conv2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # conv3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        vgg_layers = [layer for layer in vgg16.features if isinstance(layer, nn.Conv2d)]
        conv_layers = [layer for layer in self.feature_extractor if isinstance(layer, nn.Conv2d)]

        for i in range(len(conv_layers)-1):
            conv_layers[i+1].weight.data = vgg_layers[i+1].weight.data
            conv_layers[i+1].bias.data = vgg_layers[i+1].bias.data

        self.adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, feature_dim),
            nn.ReLU()
        )

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.to_poincare = ToPoincare(c=c)
        self.hyp_distance = HyperbolicDistanceLayer(c=c)
        
    def extract_blocks_with_weights(self, img):
        B, C, H, W = img.shape
        blocks = []
        weights = []
        
        for i in range(0, H-self.block_size+1, self.block_size):
            for j in range(0, W-self.block_size+1, self.block_size):
                block = img[:, :, i:i+self.block_size, j:j+self.block_size]
                blocks.append(block)
                
                weight = torch.sigmoid(torch.mean(torch.abs(
                    kornia.filters.spatial_gradient(block)
                )))
                weights.append(weight)   
        blocks = blocks[:self.num_blocks]
        weights = torch.stack(weights[:self.num_blocks])
        return blocks, weights
    
    def get_hyperbolic_features(self, img):
        blocks, weights = self.extract_blocks_with_weights(img)
        features = []
        
        for block in blocks:
            feat = self.feature_extractor(block)
            feat = self.adapter(feat)

            feat_hyp = self.to_poincare(feat)
            features.append(feat_hyp)
            
        return torch.stack(features, dim=1), weights  # [B, num_blocks, feature_dim], [num_blocks]
    
    def forward(self, rgb, aligned_ir):
        rgb_features, rgb_weights = self.get_hyperbolic_features(rgb)
        ir_features, ir_weights = self.get_hyperbolic_features(aligned_ir)

        loss = 0
        for i in range(self.num_blocks):
            rgb_feat = rgb_features[:, i, :]  # [B, feature_dim]
            ir_feat = ir_features[:, i, :]    # [B, feature_dim]

            dist = self.hyp_distance(rgb_feat, ir_feat)  

            weight = (rgb_weights[i] + ir_weights[i]) / 2

            block_loss = -torch.log(torch.sigmoid(-dist)) * weight
            loss += block_loss.mean()
            
        return loss / self.num_blocks
