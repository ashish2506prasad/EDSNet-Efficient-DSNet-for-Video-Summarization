import torch
from torch import nn
import torch.fft as fft
from anchor_based import anchor_helper
from helpers import bbox_helper
from modules.models import build_base_model
import torch.nn.functional as F
from modules.encoder import ClassicEncoder, LocalGlobalEncoder
import numpy as np
from anchor_based.poolings import Pooling
    
class DSNet_Original(nn.Module):
    def __init__(self, base_model, num_feature, num_hidden, anchor_scales,
                 num_head):
        super().__init__()
        assert base_model == 'attention'
        self.anchor_scales = anchor_scales
        self.num_scales = len(anchor_scales)
        self.base_model = build_base_model(base_model, num_feature, num_head)

        self.roi_poolings = [nn.AvgPool1d(scale, stride=1, padding=scale // 2)
                             for scale in anchor_scales]
        self.layer_norm = nn.LayerNorm(num_feature)
        self.fc1 = nn.Sequential(
            nn.Linear(num_feature, num_hidden),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.LayerNorm(num_hidden)
        )
        self.fc_cls = nn.Linear(num_hidden, 1)
        self.fc_loc = nn.Linear(num_hidden, 2)

    def forward(self, x):
        _, seq_len, _ = x.shape
        out = self.base_model(x)
        out = out + x
        out = self.layer_norm(out)  # (batch, seq_len, num_feature)
        out = out.transpose(2, 1)  # (batch, num_feature, seq_len)
        pool_results = [roi_pooling(out) for roi_pooling in self.roi_poolings]  # [(batch, num_feature, 1)] * num_scales
        out = torch.cat(pool_results, dim=0).permute(2, 0, 1)[:-1]   # (seq_len, num_scales, num_feature)
        out = self.fc1(out)  # (seq_len, num_scales, num_hidden)

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

class DSNet(nn.Module):
    def __init__(self, base_model, num_feature, num_hidden, anchor_scales,
                 num_head, fc_depth=5, orientation='paper', pooling_type = 'fft'):
        super().__init__()
        if type(anchor_scales) == int:
            anchor_scales = [anchor_scales]
        self.anchor_scales = anchor_scales
        self.num_scales = len(anchor_scales)
        self.base_model = build_base_model(base_model, num_feature, num_head, orientation)

        self.pooling_type = pooling_type

        if pooling_type == 'roi':
            self.poolings = [nn.AvgPool1d(scale, stride=1, padding=scale//2)
                                for scale in anchor_scales]
        else:
            self.poolings = Pooling(anchor_scales, pooling_type)
            self.fc_pooling = nn.Sequential(nn.Linear(num_hidden*4 , num_hidden), nn.ReLU())
                                                      
        self.layer_norm = nn.LayerNorm(num_feature)
        self.fc1 = nn.Linear(num_feature, num_hidden)
        self.fc_block = nn.Sequential(nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LayerNorm(num_hidden)
            )
        self.fc = nn.ModuleList([self.fc_block for i in range(fc_depth)])
        self.fc_cls = nn.Sequential(nn.Linear(num_hidden, num_hidden), nn.ReLU(), nn.Dropout(0.5), nn.Linear(num_hidden, 1))
        self.fc_loc = nn.Sequential(nn.Linear(num_hidden, num_hidden), nn.ReLU(), nn.Dropout(0.5), nn.Linear(num_hidden, 2))

    def forward(self, x):
        _, seq_len, _ = x.shape
        out = self.base_model(x)
        out = out + x
        out = self.fc1(self.layer_norm(out))
        for fc in self.fc:
                out = fc(out)  # (1, seq_len, num_feature)

        if self.pooling_type == 'roi':
            out = out.transpose(2, 1)  # (1, num_hidden, seq_len)
            pool_results = [roi_pooling(out) for roi_pooling in self.poolings]  # [torch.tensor(1, num_hidden, seq_len), ...]
            out = torch.cat(pool_results, dim=0).permute(2, 0, 1)[:-1]  # (seq_len, num_scales, num_hidden)
            pred_cls = self.fc_cls(out).sigmoid().view(seq_len, self.num_scales)  # (seq_len, num_scales)
            pred_loc = self.fc_loc(out).view(seq_len, self.num_scales, 2) # (seq_len, num_scales, 2)

        elif self.pooling_type == 'fft':
            pool_results = self.poolings(out) # [(seq_len, num_hidden, scale), ...]
            pool_results = torch.stack(pool_results, dim=0)  # (num_scale, seq_len, 4, num_hidden)
            coarse_pooling = pool_results.mean(dim=2)  # (num_scale, seq_len, num_hidden)
            coarse_pooling = coarse_pooling.permute(1, 0, 2)  # (seq_len, num_scale, num_hidden)
            fine_pooling = self.fc_pooling(pool_results.view(seq_len, self.num_scales, -1))
            pred_cls = self.fc_cls(coarse_pooling).sigmoid().view(seq_len, self.num_scales)
            pred_loc = self.fc_loc(fine_pooling).view(seq_len, self.num_scales, 2)

        elif self.pooling_type == 'dwt':
            pool_results = self.poolings(out) # [(seq_len, num_hidden, scale), ...]
            pool_results = torch.stack(pool_results, dim=0)  # (num_scale, seq_len, 4, num_hidden)
            coarse_pooling = pool_results.mean(dim=2)  # (num_scale, seq_len, num_hidden)
            coarse_pooling = coarse_pooling.permute(1, 0, 2)  # (seq_len, num_scale, num_hidden)
            fine_pooling = self.fc_pooling(pool_results.view(seq_len, self.num_scales, -1))
            pred_cls = self.fc_cls(coarse_pooling).sigmoid().view(seq_len, self.num_scales)
            pred_loc = self.fc_loc(fine_pooling).view(seq_len, self.num_scales, 2)


        elif self.pooling_type == 'flat-pooling':
            pool_results = self.poolings(out)
            pool_results = torch.stack(pool_results, dim=0)
            coarse_pooling = pool_results.mean(dim=2)
            coarse_pooling = coarse_pooling.permute(1, 0, 2)
            fine_pooling = self.fc_pooling(pool_results.view(seq_len, self.num_scales, -1))
            pred_cls = self.fc_cls(coarse_pooling).sigmoid().view(seq_len, self.num_scales)
            pred_loc = self.fc_loc(fine_pooling).view(seq_len, self.num_scales, 2)

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
                 num_head, fc_depth, attention_depth, orientation='paper'):
        super().__init__()
        self.anchor_scales = anchor_scales
        self.num_scales = len(anchor_scales)
        self.base_model1 = build_base_model(base_type=base_model, num_feature=num_feature, num_head=num_head//2, orientation=orientation)
        self.base_model2 = build_base_model(base_model, num_feature, num_head, orientation)

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

        self.fc_cls = nn.Sequential(nn.Linear(num_hidden, num_hidden), nn.ReLU(), nn.Dropout(0.5), nn.Linear(num_hidden, 1))
        self.fc_loc = nn.Sequential(nn.Linear(num_hidden, num_hidden), nn.ReLU(), nn.Dropout(0.5), nn.Linear(num_hidden, 2))

    def forward(self, x):
        _, seq_len, _ = x.shape
        x = self.base_model2(x + self.base_model1(x)) + x
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
        

class DSNet_MultiAttention(nn.Module):
    def __init__(self, base_model, num_feature, num_hidden, anchor_scales,
                 num_head, fc_depth, orientation='paper'):
        super().__init__()
        self.anchor_scales = anchor_scales
        self.num_scales = len(anchor_scales)
        # self.base_model1 = build_base_model(base_model, num_feature, num_head)
        self.multiattentionblock = LocalGlobalEncoder(base_model,  orientation, num_feature, num_head=num_head, num_segments=4, local_attention_head = 2)

        self.roi_poolings = [nn.AvgPool1d(scale, stride=1, padding=scale // 2)
                             for scale in anchor_scales]
        
        self.fc1 = nn.Linear(num_feature, num_hidden)
        self.fc_block = nn.Sequential(nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LayerNorm(num_hidden))
        self.fc = nn.ModuleList([self.fc_block for i in range(fc_depth)])

        self.fc_cls = nn.Sequential(nn.Linear(num_hidden, num_hidden), nn.ReLU(), nn.Linear(num_hidden, 1))
        self.fc_loc = nn.Sequential(nn.Linear(num_hidden, num_hidden), nn.ReLU(), nn.Linear(num_hidden, 2))

    def forward(self, x):
        _, seq_len, _ = x.shape
        # global_attention = self.base_model1(x)
        attention = self.multiattentionblock(x)
      
        out = x + attention
        out = self.fc1(out)
        for fc in self.fc:
            out = fc(out)

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
    

class DSNetMotionFeatures(nn.Module):
    def __init__(self, base_model, num_feature, num_hidden, anchor_scales,
                 num_head, attention_depth, encoder_type='classic'):
        super().__init__()
        self.anchor_scales = anchor_scales
        self.num_scales = len(anchor_scales)
        self.base_model = base_model
        
        if encoder_type == 'classic':
            self.encoder = ClassicEncoder(base_model, num_feature, num_head, attention_depth)
        elif encoder_type == 'local_global':
            self.encoder = LocalGlobalEncoder(base_model, num_feature, num_head, attention_depth)

        decoder_layer = nn.TransformerDecoderLayer(d_model=1024 , nhead=8, batch_first=True, dim_feedforward=num_feature)
        self.multiheadcrossattention = nn.TransformerDecoder(decoder_layer, num_layers=attention_depth)

        self.roi_poolings = [nn.AvgPool1d(scale, stride=1, padding=scale // 2)
                             for scale in anchor_scales]

        self.layer_norm = nn.LayerNorm(num_feature)
        self.fc_block = nn.Sequential(nn.Linear(num_feature, num_hidden),
                                      nn.Linear(num_hidden, num_hidden),
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.LayerNorm(num_hidden)
                                      )
        
        self.fc_cls = nn.Linear(num_hidden, 1)
        self.fc_loc = nn.Linear(num_hidden, 2)

    def forward(self, x, motion_features):
        _, seq_len, _ = x.shape
        out = self.encoder(x)
        out = out + self.multiheadcrossattention(memory=motion_features, tgt=out)

        out = self.fc_block(self.layer_norm(out))

        out = out.transpose(2, 1)
        pool_results = [roi_pooling(out) for roi_pooling in self.roi_poolings]
        out = torch.cat(pool_results, dim=0).permute(2, 0, 1)[:-1]

        pred_cls = self.fc_cls(out).sigmoid().view(seq_len, self.num_scales)
        pred_loc = self.fc_loc(out).view(seq_len, self.num_scales, 2)

        return pred_cls, pred_loc

    def predict(self, seq, motion_features):
        seq_len = seq.shape[1]
        pred_cls, pred_loc = self(seq, motion_features)

        pred_cls = pred_cls.cpu().numpy().reshape(-1)
        pred_loc = pred_loc.cpu().numpy().reshape((-1, 2))

        anchors = anchor_helper.get_anchors(seq_len, self.anchor_scales)
        anchors = anchors.reshape((-1, 2))

        pred_bboxes = anchor_helper.offset2bbox(pred_loc, anchors)
        pred_bboxes = bbox_helper.cw2lr(pred_bboxes)

        return pred_cls, pred_bboxes