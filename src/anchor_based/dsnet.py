import torch
from torch import nn

from anchor_based import anchor_helper
from helpers import bbox_helper
from modules.models import build_base_model
import torch.nn.functional as F
from modules.encoder import ClassicEncoder, LocalGlobalEncoder

class DSNet_Original(nn.Module):
    def __init__(self, base_model, num_feature, num_hidden, anchor_scales,
                 num_head):
        super().__init__()
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

class DSNet(nn.Module):
    def __init__(self, base_model, num_feature, num_hidden, anchor_scales,
                 num_head, fc_depth=5, orientation='paper'):
        super().__init__()
        self.anchor_scales = anchor_scales
        self.num_scales = len(anchor_scales)
        self.base_model = build_base_model(base_model, num_feature, num_head, orientation)

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
        

class DSNet_MultiAttention(nn.Module):
    def __init__(self, base_model, num_feature, num_hidden, anchor_scales,
                 num_head, fc_depth, orientation='paper'):
        super().__init__()
        self.anchor_scales = anchor_scales
        self.num_scales = len(anchor_scales)
        # self.base_model1 = build_base_model(base_model, num_feature, num_head)
        self.multiattentionblock = LocalGlobalEncoder(num_feature, base_model, orientation, num_segments=4, num_head=num_head, local_attention_head = 2)

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


