# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcls.evaluation.metrics import Accuracy
from mmcls.registry import MODELS
from mmcls.structures import ClsDataSample
from .base_head import BaseHead
import numpy as np
from lifelines.utils import concordance_index 
import random
def coxph_loss(y_pred, y_time, y_event):

    return -censored_likelihood

def c_index(y_pred, y_time, y_event):
    time = y_time
    event = y_event
    return concordance_index(time.cpu().detach().numpy(), -np.exp(y_pred.cpu().detach().numpy()), event.cpu().detach().numpy())
@MODELS.register_module()
class SurvivalHead(BaseHead):
    """Classification head.

    Args:
        loss (dict): Config of classification loss. Defaults to
            ``dict(type='CrossEntropyLoss', loss_weight=1.0)``.
        topk (int | Tuple[int]): Top-k accuracy. Defaults to ``(1, )``.
        cal_acc (bool): Whether to calculate accuracy during training.
            If you use batch augmentations like Mixup and CutMix during
            training, it is pointless to calculate accuracy.
            Defaults to False.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to None.
    """

    def __init__(self,
                 loss: dict = dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk: Union[int, Tuple[int]] = (1, ),
                 cal_acc: bool = False,
                 init_cfg: Optional[dict] = None):
        super(SurvivalHead, self).__init__(init_cfg=init_cfg)

        self.topk = topk
        if not isinstance(loss, nn.Module):
            loss = MODELS.build(loss)
        self.loss_module = loss
        self.cal_acc = cal_acc

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage. In ``ClsHead``, we just obtain the feature
        of the last stage.
        """
        # The ClsHead doesn't have other module, just return after unpacking.
        return feats[-1]

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        # The ClsHead doesn't have the final classification head,
        # just return the unpacked inputs.
        return pre_logits

    def loss(self, feats: Tuple[torch.Tensor],
             data_samples: List[ClsDataSample], **kwargs) -> dict:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[ClsDataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # The part can be traced by torch.fx
        
        cls_score,y_pred = self(feats)
        
        y_time = [i.y_time for i in data_samples]
        y_event = [i.y_event for i in data_samples]
        
        
        y_time = torch.tensor(y_time).to(cls_score.device)
        y_event = torch.tensor(y_event).to(cls_score.device)
        y_pred = y_pred.squeeze()
        # The part can not be traced by torch.fx
        # coxph_loss()
        

        losses = self._get_loss(cls_score, data_samples,y_pred, y_time, y_event, **kwargs)

        return losses

    def _get_loss(self, cls_score: torch.Tensor,
                  data_samples: List[ClsDataSample],y_pred, y_time, y_event, **kwargs):
        """Unpack data samples and compute loss."""
        # Unpack data samples and pack targets
        
        if 'score' in data_samples[0].gt_label:
            # Batch augmentation may convert labels to one-hot format scores.
            target = torch.stack([i.gt_label.score for i in data_samples])
        else:
            target = torch.cat([i.gt_label.label for i in data_samples])

        # compute loss
        losses = dict()
        # loss = self.loss_module(
        #     cls_score, target, avg_factor=cls_score.size(0), **kwargs)
        # losses['loss'] = loss
        ####Cox loss
        time = y_time
        event = y_event
        sort_time = torch.argsort(time, 0, descending=True)
        event = torch.gather(event, 0, sort_time)
        risk = torch.gather(y_pred, 0, sort_time)
        exp_risk = torch.exp(risk)
        log_risk = torch.log(torch.cumsum(exp_risk, 0))
        censored_likelihood = (risk - log_risk) * event
        censored_likelihood = torch.sum(censored_likelihood)
        censored_likelihood = censored_likelihood / y_time.shape[0]
        losses['loss'] = -censored_likelihood

        
        # compute accuracy
        if self.cal_acc:

            assert target.ndim == 1, 'If you enable batch augmentation ' \
                'like mixup during training, `cal_acc` is pointless.'
            acc = Accuracy.calculate(cls_score, target, topk=self.topk)
            losses.update(
                {f'accuracy_top-{k}': a
                 for k, a in zip(self.topk, acc)})
            losses.update(
                {f'C-index':C_index})

        return losses

    def predict(
        self,
        feats: Tuple[torch.Tensor],
        data_samples: List[Union[ClsDataSample, None]] = None
    ) -> List[ClsDataSample]:
        """Inference without augmentation.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[ClsDataSample | None], optional): The annotation
                data of every samples. If not None, set ``pred_label`` of
                the input data samples. Defaults to None.

        Returns:
            List[ClsDataSample]: A list of data samples which contains the
            predicted results.
        """
        # The part can be traced by torch.fx
        cls_score,y_pred = self(feats)
        y_pred = y_pred.squeeze()
        
        # The part can not be traced by torch.fx
        predictions = self._get_predictions(cls_score,y_pred, data_samples)

        return predictions

    def _get_predictions(self, cls_score,y_pred, data_samples):
        """Post-process the output of head.

        Including softmax and set ``pred_label`` of data samples.
        """
        
        pred_scores = F.softmax(cls_score, dim=1)
        pred_labels = pred_scores.argmax(dim=1, keepdim=True).detach()
        
        out_data_samples = []
        if data_samples is None:
            data_samples = [None for _ in range(pred_scores.size(0))]

        for data_sample, score, label,y_pre in zip(data_samples, pred_scores,
                                             pred_labels,y_pred):
            if data_sample is None:
                data_sample = ClsDataSample()

            data_sample.set_pred_score(score).set_pred_label(label)
            data_sample.set_field(
                name='y_pred',
                value=y_pre,
                field_type='metainfo')
            
            out_data_samples.append(data_sample)
            
        return out_data_samples
