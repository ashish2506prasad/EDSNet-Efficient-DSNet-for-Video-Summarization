import torch
from torch import nn

from anchor_based import anchor_helper
from helpers import bbox_helper
from modules.models import build_base_model
import torch.nn.functional as F
import math


class DSNet(nn.Module):
    def __init__(self, base_model, num_feature, num_hidden, anchor_scales,
                 num_head, fc_depth):
        super().__init__()
        self.anchor_scales = anchor_scales
        self.num_scales = len(anchor_scales)
        self.base_model = build_base_model(base_model, num_feature, num_head)

        self.roi_poolings = [nn.AvgPool1d(scale, stride=1, padding=scale // 2)
                             for scale in anchor_scales]

        self.layer_norm = nn.LayerNorm(num_feature)
        self.fc1 = nn.Linear(num_feature, num_hidden)
        self.fc_block = nn.Sequential(nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LayerNorm(num_hidden)
            )
        self.fc = nn.ModuleList([self.fc_block for i in range(fc_depth)])
        
        self.fc_cls = nn.Linear(num_hidden, 1)
        self.fc_loc = nn.Linear(num_hidden, 2)

    def forward(self, x):
        _, seq_len, _ = x.shape
        if self.base_model == 'fourier' or self.base_model == 'fast-fourier':
            out, _ = self.base_model(x)
        else:
            out = self.base_model(x)
        out = out + x
        out = self.fc1(self.layer_norm(out))
        for fc in self.fc:
            out = fc(out)

        out = out.transpose(2, 1)
        pool_results = [roi_pooling(out) for roi_pooling in self.roi_poolings]
        out = torch.cat(pool_results, dim=0).permute(2, 0, 1)[:-1]

        pred_cls = self.fc_cls(out).sigmoid().view(seq_len, self.num_scales)
        pred_loc = self.fc_loc(out).view(seq_len, self.num_scales, 2)

        return pred_cls, pred_loc

    def predict(self, seq):
        seq_len = seq.shape[1]
        pred_cls, pred_loc = self(seq)

        pred_cls = pred_cls.cpu().numpy().reshape(-1)
        pred_loc = pred_loc.cpu().numpy().reshape((-1, 2))

        anchors = anchor_helper.get_anchors(seq_len, self.anchor_scales)
        anchors = anchors.reshape((-1, 2))

        pred_bboxes = anchor_helper.offset2bbox(pred_loc, anchors)
        pred_bboxes = bbox_helper.cw2lr(pred_bboxes)

        return pred_cls, pred_bboxes


class DSNet_DeepAttention(nn.Module):
    def __init__(self, base_model, num_feature, num_hidden, anchor_scales,
                 num_head, fc_depth, attention_depth):
        super().__init__()
        self.anchor_scales = anchor_scales
        self.num_scales = len(anchor_scales)
        self.base_model1 = build_base_model(base_model, num_feature, num_head//2)
        self.base_model2 = build_base_model(base_model, num_feature, num_head)

        self.roi_poolings = [nn.AvgPool1d(scale, stride=1, padding=scale // 2)
                             for scale in anchor_scales]

        self.layer_norm = nn.LayerNorm(num_feature)

        self.fc1 = nn.Linear(num_feature, num_hidden)

        self.fc_block = nn.Sequential(nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LayerNorm(num_hidden))
        self.fc = nn.ModuleList([self.fc_block for i in range(fc_depth)])

        self.attention_block = nn.ModuleList([self.base_model1 for i in range(attention_depth-1)])

        self.fc_cls = nn.Linear(num_hidden, 1)
        self.fc_loc = nn.Linear(num_hidden, 2)

    def forward(self, x):
        _, seq_len, _ = x.shape
        x = self.base_model2(x)
        for attention_layer in self.attention_block:
            out = attention_layer(x)
            x = x + out
        out = x

        out = self.fc1(out)
        for fc in self.fc:
            out = fc(out)

        out = out.transpose(2, 1)
        pool_results = [roi_pooling(out) for roi_pooling in self.roi_poolings]
        out = torch.cat(pool_results, dim=0).permute(2, 0, 1)[:-1]

        pred_cls = self.fc_cls(out).sigmoid().view(seq_len, self.num_scales)
        pred_loc = self.fc_loc(out).view(seq_len, self.num_scales, 2)

        return pred_cls, pred_loc

    def predict(self, seq):
        seq_len = seq.shape[1]
        pred_cls, pred_loc = self(seq)

        pred_cls = pred_cls.cpu().numpy().reshape(-1)
        pred_loc = pred_loc.cpu().numpy().reshape((-1, 2))

        anchors = anchor_helper.get_anchors(seq_len, self.anchor_scales)
        anchors = anchors.reshape((-1, 2))

        pred_bboxes = anchor_helper.offset2bbox(pred_loc, anchors)
        pred_bboxes = bbox_helper.cw2lr(pred_bboxes)

        return pred_cls, pred_bboxes


class DSNetTriangularAttention(nn.Module):
    def __init__(self, base_model, num_feature, num_hidden, anchor_scales,
                 num_head):
        super().__init__()
        self.anchor_scales = anchor_scales
        self.num_scales = len(anchor_scales)
        self.base_model1 = build_base_model(base_model, num_feature, num_head)
        self.base_model2 = build_base_model(base_model, num_feature//2, num_head*2)
        self.base_model3 = build_base_model(base_model, num_feature//4, num_head*4)
        self.base_model4 = build_base_model(base_model, num_feature//8, num_head*8)

        self.roi_poolings = [nn.AvgPool1d(scale, stride=1, padding=scale // 2)
                             for scale in anchor_scales]

        self.layer_norm1 = nn.LayerNorm(num_feature)
        self.layer_norm2 = nn.LayerNorm(num_feature//2)
        self.layer_norm3 = nn.LayerNorm(num_feature//4)
        self.layer_norm4 = nn.LayerNorm(num_feature//8)

        self.fc1 = nn.Sequential(nn.Linear(num_feature, num_feature//2), nn.Dropout(0.5), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(num_feature//2, num_feature//4), nn.Dropout(0.5), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(num_feature//4, num_feature//8), nn.Dropout(0.5), nn.ReLU())
        self.fc4 = nn.Sequential(nn.Linear(num_feature//8, num_hidden), nn.Dropout(0.5), nn.ReLU())

        self.fc_cls = nn.Linear(num_hidden, 1)
        self.fc_loc = nn.Linear(num_hidden, 2)

    def forward(self, x):
        _, seq_len, _ = x.shape
        # print(x.shape)
        out = self.fc1(self.layer_norm1(self.base_model1(x) + x))
        # print(out.shape)
        out = self.fc2(self.layer_norm2(self.base_model2(out) + out))
        # print(out.shape)
        out = self.fc3(self.layer_norm3(self.base_model3(out) + out))
        # print(out.shape)
        out = self.fc4(self.layer_norm4(self.base_model4(out) + out)) 
        # print(out.shape)

        out = out.transpose(2, 1)
        pool_results = [roi_pooling(out) for roi_pooling in self.roi_poolings]
        out = torch.cat(pool_results, dim=0).permute(2, 0, 1)[:-1]

        pred_cls = self.fc_cls(out).sigmoid().view(seq_len, self.num_scales)
        pred_loc = self.fc_loc(out).view(seq_len, self.num_scales, 2)

        return pred_cls, pred_loc

    def predict(self, seq):
        seq_len = seq.shape[1]
        pred_cls, pred_loc = self(seq)

        pred_cls = pred_cls.cpu().numpy().reshape(-1)
        pred_loc = pred_loc.cpu().numpy().reshape((-1, 2))

        anchors = anchor_helper.get_anchors(seq_len, self.anchor_scales)
        anchors = anchors.reshape((-1, 2))

        pred_bboxes = anchor_helper.offset2bbox(pred_loc, anchors)
        pred_bboxes = bbox_helper.cw2lr(pred_bboxes)

        return pred_cls, pred_bboxes


class MultiAttention(nn.Module):
    def __init__(self, num_feature, base_model, num_segments=5 , num_head=8, local_attention_head=4):
        super(MultiAttention, self).__init__()

        self.num_segments = num_segments
        self.global_attention = build_base_model(base_model, num_feature, num_head)
        self.layer_norm = nn.LayerNorm(num_feature)
        self.fc = nn.Sequential(nn.Linear(num_feature, num_feature),
                                nn.ReLU())

        self.num_segments = num_segments
        if self.num_segments is not None:
            assert self.num_segments >= 2, "num_segments must be None or 2+"
            self.local_attention = nn.ModuleList()
            for _ in range(self.num_segments):
                # Local Attention, considering differences among the same segment with reduce hidden size
                self.local_attention.append(build_base_model(base_model, num_feature, num_head=local_attention_head))

    def forward(self, x):
        """ Computes multi-attention of different segments and fuses the results
        """
        # print(x.shape)
        weighted_value = self.fc(self.global_attention(x))  # global attention
        # print(weighted_value.shape)

        if self.num_segments is not None :
            segment_size = math.ceil(x.shape[-2] / self.num_segments)
            # print(segment_size)
            for segment in range(self.num_segments):
                left_pos = segment * segment_size
                right_pos = (segment + 1) * segment_size
                local_x = x[:,left_pos:right_pos,:]
                weighted_local_value = self.fc(self.local_attention[segment](local_x))  # local attentions
                # print(weighted_local_value.shape)

                # Normalize the features vectors
                weighted_value[left_pos:right_pos] = F.normalize(weighted_value[left_pos:right_pos].clone(), p=2, dim=1)
                weighted_local_value = F.normalize(weighted_local_value, p=2, dim=1)
                # print(weighted_value[left_pos:right_pos].shape)
                weighted_value[:,left_pos:right_pos] += weighted_local_value

        return weighted_value
        

class DSNet_MultiAttention(nn.Module):
    def __init__(self, base_model, num_feature, num_hidden, anchor_scales,
                 num_head):
        super().__init__()
        self.anchor_scales = anchor_scales
        self.num_scales = len(anchor_scales)
        # self.base_model1 = build_base_model(base_model, num_feature, num_head)
        self.multiattentionblock = MultiAttention(num_feature, base_model, num_segments=4, num_head=num_head, local_attention_head = 2)

        self.roi_poolings = [nn.AvgPool1d(scale, stride=1, padding=scale // 2)
                             for scale in anchor_scales]

        self.layer_norm = nn.LayerNorm(num_feature)
        self.fc1 = nn.Sequential(
            nn.Linear(num_feature, num_hidden),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.LayerNorm(num_hidden),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LayerNorm(num_hidden),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LayerNorm(num_hidden)
        )
        self.fc_cls = nn.Sequential(nn.Linear(num_hidden, num_hidden), nn.ReLU(), nn.Linear(num_hidden, 1))
        self.fc_loc = nn.Sequential(nn.Linear(num_hidden, num_hidden), nn.ReLU(), nn.Linear(num_hidden, 2))

    def forward(self, x):
        _, seq_len, _ = x.shape
        # global_attention = self.base_model1(x)
        attention = self.multiattentionblock(x)
      
        out = x + attention
        out = self.layer_norm(out)

        out = out.transpose(2, 1)
        pool_results = [roi_pooling(out) for roi_pooling in self.roi_poolings]
        out = torch.cat(pool_results, dim=0).permute(2, 0, 1)[:-1]

        out = self.fc1(out)

        pred_cls = self.fc_cls(out).sigmoid().view(seq_len, self.num_scales)
        pred_loc = self.fc_loc(out).view(seq_len, self.num_scales, 2)

        return pred_cls, pred_loc

    def predict(self, seq):
        seq_len = seq.shape[1]
        pred_cls, pred_loc = self(seq)

        pred_cls = pred_cls.cpu().numpy().reshape(-1)
        pred_loc = pred_loc.cpu().numpy().reshape((-1, 2))

        anchors = anchor_helper.get_anchors(seq_len, self.anchor_scales)
        anchors = anchors.reshape((-1, 2))

        pred_bboxes = anchor_helper.offset2bbox(pred_loc, anchors)
        pred_bboxes = bbox_helper.cw2lr(pred_bboxes)

        return pred_cls, pred_bboxes


