import torch
import torch.nn as nn
import torch.nn.functional as F
from hyptorch.pmath import dist_matrix, project, expmap0, logmap0
from hyptorch.nn import HyperbolicDistanceLayer,ToPoincare

class HyperbolicEdgeLoss(nn.Module):
    def __init__(self, c=1.0, epsilon=1e-6):
        super().__init__()
        self.c = c
        self.epsilon = epsilon
        self.hyp_dist = HyperbolicDistanceLayer(c=c)
        self.to_hyperbolic = ToPoincare(c=c, train_c=False)

        self.sobel_x = torch.tensor([[-1, 0, 1], 
                                   [-2, 0, 2], 
                                   [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], 
                                   [0, 0, 0], 
                                   [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        
    def to(self, device):
        super().to(device)
        self.sobel_x = self.sobel_x.to(device)
        self.sobel_y = self.sobel_y.to(device)
        return self
        
    def compute_gradients(self, x):
        if x.shape[1] > 1:
            x = torch.mean(x, dim=1, keepdim=True)

        grad_x = F.conv2d(x, self.sobel_x.expand(1, 1, 3, 3).to(x.device), padding=1)
        grad_y = F.conv2d(x, self.sobel_y.expand(1, 1, 3, 3).to(x.device), padding=1)
        
        return grad_x, grad_y
    
    def prepare_for_hyperbolic(self, grad_x, grad_y):
        B, C, H, W = grad_x.shape

        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + self.epsilon)

        grad_flat = grad_magnitude.view(B, -1)  # (B, H*W)
        grad_norm = F.normalize(grad_flat, p=2, dim=1)

        grad_hyp = project(expmap0(grad_norm, c=self.c), c=self.c)
        
        return grad_hyp
        
    def forward(self, x, y, is_hpy=True):
        grad_x1, grad_y1 = self.compute_gradients(x)
        grad_x2, grad_y2 = self.compute_gradients(y)
        
        if is_hpy:
            grad1_hyp = self.prepare_for_hyperbolic(grad_x1, grad_y1)
            grad2_hyp = self.prepare_for_hyperbolic(grad_x2, grad_y2)

            dist = self.hyp_dist(grad1_hyp, grad2_hyp)

            loss = torch.sqrt(dist**2 + self.epsilon**2).mean()
        
        else:
            grad_diff = torch.abs(grad_x1 - grad_x2) + torch.abs(grad_y1 - grad_y2)
            loss = grad_diff.mean()
        
        return loss

class CharbonnierHyperbolicLoss(nn.Module):
    def __init__(self, c=1.0, epsilon=1e-6):
        super().__init__()
        self.edge_loss = HyperbolicEdgeLoss(c=c, epsilon=epsilon)
        
    def forward(self, x, y, is_hpy=True):
        return self.edge_loss(x, y, is_hpy)