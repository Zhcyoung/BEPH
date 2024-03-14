# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.registry import MODELS
from .cls_head import ClsHead
from .Attn_Net_Gated import Attn_Net_Gated

@MODELS.register_module()
class Linear_head_subtyping(ClsHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss. Defaults to
            ``dict(type='CrossEntropyLoss', loss_weight=1.0)``.
        topk (int | Tuple[int]): Top-k accuracy. Defaults to ``(1, )``.
        cal_acc (bool): Whether to calculate accuracy during training.
            If you use batch augmentations like Mixup and CutMix during
            training, it is pointless to calculate accuracy.
            Defaults to False.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to ``dict(type='Normal', layer='Linear', std=0.01)``.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 init_cfg: Optional[dict] = dict(
                     type='Normal', layer='Linear', std=0.01),
                 **kwargs):
        super(Linear_head_subtyping, self).__init__(init_cfg=init_cfg, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.size_dict_path = {"small": [384, 192, 192], "big": [1024, 512, 384]}
        size = self.size_dict_path['small']

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        # self.fc = nn.Linear(self.in_channels, self.num_classes)
        self.global_phi = nn.Sequential(nn.Linear(self.in_channels, 192), nn.ReLU(), nn.Dropout(0.25))
        self.global_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=192, nhead=3, dim_feedforward=192, dropout=0.25, activation='relu'
        ), 
            num_layers=2
        )
        self.global_attn_pool = Attn_Net_Gated(L=size[1], D=size[1], dropout=0.25, n_classes=1)
        self.global_rho = nn.Sequential(*[nn.Linear(size[1], size[1]), nn.ReLU(), nn.Dropout(0.25)])

        self.classifier = nn.Linear(size[1], self.num_classes)
    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage. In ``Linear_head_subtyping``, we just obtain the
        feature of the last stage.
        """
        # The Linear_head_subtyping doesn't have other module, just return after
        # unpacking.
        return feats[-1]

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        # The final classification head.
        h_4096 = self.global_phi(pre_logits)
        h_4096 = self.global_transformer(h_4096.unsqueeze(1)).squeeze(1)
        A_4096, h_4096 = self.global_attn_pool(h_4096)  
        A_4096 = torch.transpose(A_4096, 1, 0)
        A_4096 = F.softmax(A_4096, dim=1) 
#         h_path = A_4096 * h_4096
        
#         # h_path = torch.mm(A_4096, h_4096)
#         h_WSI = self.global_rho(h_path)
#         cls_score = self.fc(h_WSI)

        h_path = torch.mm(A_4096, h_4096)
        
        h_path = self.global_rho(h_path)
        
        logits = self.classifier(h_path)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        Y_prob = torch.tile(Y_prob, (224, 1))
        return Y_prob
