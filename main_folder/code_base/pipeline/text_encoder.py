import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers as tfe
from transformers import AutoModel, AutoConfig
from main_folder.code_base.utils import ArcMarginProduct, CurricularFace


class TextEncoder(nn.Module):
    def __init__(
        self,
        num_classes,
        embed_size=1024,
        max_seq_length=35,
        backbone=None,
        dropout=0.5,
        scale=30.0,
        margin=0.5,
        final_layer="arcface",
        device="cuda",
        eval_model=False,
        alpha=0.0,
    ):
        super().__init__()
        self.backbone_name = backbone
        if eval_model:
            self.config = AutoConfig.from_pretrained(backbone)
            self.backbone = AutoModel.from_config(self.config)
        else:
            self.backbone = AutoModel.from_pretrained(backbone)
        self.out_features = num_classes
        self.embed_size = embed_size
        self.scale = scale
        self.margin = margin
        self.device = device

        if final_layer == "arcface":
            self.final = ArcMarginProduct(
                in_features=self.embed_size,
                out_features=self.out_features,
                s=self.scale,
                m=self.margin,
                device=self.device,
                alpha=alpha,
            )

        if final_layer == "currface":
            self.final = CurricularFace(
                in_features=self.embed_size,
                out_features=self.out_features,
                s=self.scale,
                m=self.margin,
            )

        self.fc = nn.Linear(self.backbone.config.hidden_size, self.embed_size)
        self.pool = nn.AvgPool1d(kernel_size=max_seq_length)
        self.bn = nn.BatchNorm1d(self.embed_size)

    def forward(self, input_ids, attention_mask, labels=None):
        features = self.backbone(
            input_ids, attention_mask=attention_mask
        ).last_hidden_state
        features = self.fc(features)
        features = features.transpose(1, 2)
        features = self.pool(features)
        features = features.view(features.size(0), -1)
        features = self.bn(features)
        features = F.normalize(features)
        if labels is not None:
            return self.final(features, labels)
        return features
