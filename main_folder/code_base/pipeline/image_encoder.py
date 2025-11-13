import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import ScaledStdConv2d, ScaledStdConv2dSame, BatchNormAct2d
from main_folder.code_base.utils import ArcMarginProduct, CurricularFace
from main_folder.code_base.pipeline.gempool import GeM
from main_folder.code_base.pipeline.depthconv import DepthwiseSeparableConv


class ImgEncoder(nn.Module):
    def __init__(
        self,
        num_classes,
        embed_size=1792,
        backbone=None,
        pretrained=True,
        scale=30.0,
        margin=0.5,
        alpha=0.0,
        final_layer="arcface",
        device="cuda",
        permute=False,
        p=3,
    ):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained)
        self.embed_size = embed_size  # embedding size
        self.num_classes = num_classes  # num classes
        self.margin = margin
        self.scale = scale
        self.device = device
        self.p = p

        self.final_conv = nn.Conv2d(
            self.backbone.num_features,
            self.embed_size,
            kernel_size=1,
        )

        if final_layer == "arcface":
            self.final = ArcMarginProduct(
                in_features=self.embed_size,
                out_features=self.num_classes,
                s=self.scale,
                m=self.margin,
                alpha=alpha,
                device=self.device,
            )

        if final_layer == "currface":
            self.final = CurricularFace(
                in_features=self.embed_size,
                out_features=self.num_classes,
                s=self.scale,
                m=self.margin,
                alpha=alpha,
            )

        self.gem = GeM(p=self.p)
        self.bn = nn.BatchNorm1d(self.embed_size)
        self.permute = permute

    def forward(self, x, labels=None):
        features = self.backbone.forward_features(x)
        if self.permute:
            features = torch.permute(features, (0, 3, 1, 2))
        features = self.final_conv(features)
        features = self.gem(features)
        features = features.view(features.size(0), -1)
        features = self.bn(features)
        features = F.normalize(features)
        if labels is not None:
            return self.final(features, labels)
        return features
