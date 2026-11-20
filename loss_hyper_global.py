import torch
import torch.nn as nn
import torchvision.models as models
from hyptorch.nn import HyperbolicDistanceLayer, ToPoincare


class GlobalHyperbolicLoss(nn.Module):
    def __init__(self, feature_dim: int = 256, c: float = 1.0):
        super().__init__()
        self.feature_dim = feature_dim

        vgg16 = models.vgg16(pretrained=False)
        vgg16.load_state_dict(torch.load('hyptorch/vgg16-397923af.pth'))

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        with torch.no_grad():
            self.conv1.weight.data = torch.sum(vgg16.features[0].weight.data, dim=1, keepdim=True)
        self.conv1.requires_grad_(False)

        self.feature_extractor = nn.Sequential(
            self.conv1,
            *list(vgg16.features[1:16])
        )

        vgg_layers = [layer for layer in vgg16.features[1:16] if isinstance(layer, nn.Conv2d)]
        conv_layers = [layer for layer in self.feature_extractor[1:] if isinstance(layer, nn.Conv2d)]

        for vgg_layer, conv_layer in zip(vgg_layers, conv_layers):
            conv_layer.weight.data = vgg_layer.weight.data
            conv_layer.bias.data = vgg_layer.bias.data
            conv_layer.requires_grad_(False)

        self.adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, feature_dim, bias=False),
            nn.LayerNorm(feature_dim, elementwise_affine=False),
            nn.ReLU(),
        )

        with torch.no_grad():
            nn.init.orthogonal_(self.adapter[2].weight)

        for param in self.adapter.parameters():
            param.requires_grad = False

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.to_poincare = ToPoincare(c=c)
        self.hyp_distance = HyperbolicDistanceLayer(c=c)

    def get_hyperbolic_features(self, img: torch.Tensor) -> torch.Tensor:
        feat = self.feature_extractor(img)
        feat = self.adapter(feat)
        return self.to_poincare(feat)

    def get_euclidian_features(self, img: torch.Tensor) -> torch.Tensor:
        feat = self.feature_extractor(img)
        return self.adapter(feat)

    def forward(self, rgb: torch.Tensor, aligned_ir: torch.Tensor, is_hpy: bool = True) -> torch.Tensor:
        if is_hpy:
            rgb_feat = self.get_hyperbolic_features(rgb)
            ir_feat = self.get_hyperbolic_features(aligned_ir)
            dist = self.hyp_distance(rgb_feat, ir_feat)
        else:
            rgb_feat = self.get_euclidian_features(rgb)
            ir_feat = self.get_euclidian_features(aligned_ir)
            dist = torch.sqrt(torch.sum((rgb_feat - ir_feat) ** 2))

        loss = -torch.log(torch.sigmoid(-dist)).mean()
        return loss


class Vis_GlobalHyperbolicLoss(nn.Module):
    def __init__(self, feature_dim: int = 256, c: float = 1.0):
        super().__init__()
        self.feature_dim = feature_dim

        vgg16 = models.vgg16(pretrained=False)
        vgg16.load_state_dict(torch.load('hyptorch/vgg16-397923af.pth'))

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        with torch.no_grad():
            self.conv1.weight.data = torch.sum(vgg16.features[0].weight.data, dim=1, keepdim=True)
        self.conv1.requires_grad_(False)

        self.feature_extractor = nn.Sequential(
            self.conv1,
            *list(vgg16.features[1:16])
        )

        vgg_layers = [layer for layer in vgg16.features[1:16] if isinstance(layer, nn.Conv2d)]
        conv_layers = [layer for layer in self.feature_extractor[1:] if isinstance(layer, nn.Conv2d)]

        for vgg_layer, conv_layer in zip(vgg_layers, conv_layers):
            conv_layer.weight.data = vgg_layer.weight.data
            conv_layer.bias.data = vgg_layer.bias.data
            conv_layer.requires_grad_(False)

        self.adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, feature_dim, bias=False),
            nn.LayerNorm(feature_dim, elementwise_affine=False),
            nn.ReLU(),
        )

        with torch.no_grad():
            nn.init.orthogonal_(self.adapter[2].weight)

        for param in self.adapter.parameters():
            param.requires_grad = False

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.to_poincare = ToPoincare(c=c)
        self.hyp_distance = HyperbolicDistanceLayer(c=c)

    def get_hyperbolic_features(self, img: torch.Tensor) -> torch.Tensor:
        feat = self.feature_extractor(img)
        feat = self.adapter(feat)
        return self.to_poincare(feat)

    def get_euclidian_features(self, img: torch.Tensor) -> torch.Tensor:
        feat = self.feature_extractor(img)
        return self.adapter(feat)

    def forward(self, rgb: torch.Tensor, aligned_ir: torch.Tensor, is_hpy: bool = True):
        if is_hpy:
            rgb_feat = self.get_hyperbolic_features(rgb)
            ir_feat = self.get_hyperbolic_features(aligned_ir)
            dist = self.hyp_distance(rgb_feat, ir_feat)
        else:
            rgb_feat = self.get_euclidian_features(rgb)
            ir_feat = self.get_euclidian_features(aligned_ir)
            dist = torch.sqrt(torch.sum((rgb_feat - ir_feat) ** 2))

        loss = -torch.log(torch.sigmoid(-dist)).mean()
        return rgb_feat, ir_feat, loss
