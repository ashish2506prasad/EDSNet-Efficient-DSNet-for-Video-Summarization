import torch
from torch import nn
import torch.fft as fft
from anchor_based import anchor_helper
from helpers import bbox_helper
from modules.models import build_base_model
import torch.nn.functional as F
from modules.encoder import ClassicEncoder, LocalGlobalEncoder
import numpy as np
import pywt

class Pooling(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        assert scale in [[4, 8], [4], [8]], "anchor scale must be [4, 8], [4], or [8]"

    @staticmethod
    def dwt(x, scale):
        assert scale == 8
        pooled = []
        coarse_pooling = []
        for i in range(x.shape[1]):
            end = min(i + scale//2, x.shape[1])
            start = max(0, i - scale//2 + 1)
            segment = x[:, start:end+1, :]
            if segment.shape[1] < scale:
                segment = F.pad(segment, (0, 0, 0, scale - segment.shape[1]))
            coeffs, _ = pywt.dwt(segment.cpu().numpy(), 'db1', axis = 1)
            coeffs = torch.from_numpy(coeffs).to(x.device)
            pooled.append(coeffs)
            coarse_pooling.append(coeffs.mean(dim=1))
        
        fine_pooling = torch.stack(pooled, dim=1).view(1, x.shape[1],4 * x.shape[2]).to(x.device)
        coarse_pooling = torch.stack(coarse_pooling, dim=1).to(x.device)
        return coarse_pooling, fine_pooling

    @staticmethod
    def fft(x, scale):
        assert scale == 4
        pooled = []
        coarse_pooling = []
        for i in range(x.shape[1]):
            end = min(i + scale//2, x.shape[1])
            start = max(0, i - scale//2 + 1)
            segment = x[:, start:end+1, :]
            if segment.shape[1] < scale:
                segment = F.pad(segment, (0, 0, 0, scale - segment.shape[1]))
            segment_fft = fft.fft(segment, dim=1).real  # (1, 4, num_hidden)
            coarse_pooling.append(segment_fft.mean(dim=1))  # (1, 1, num_hidden)
            pooled.append(segment_fft)
        
        coarse_pooling = torch.stack(coarse_pooling, dim=1).to(x.device)
        fine_pooling = torch.stack(pooled, dim=1).view(1, x.shape[1], scale * x.shape[2]).to(x.device)
        return coarse_pooling, fine_pooling

    def forward(self, x):
        if self.scale == [4, 8]:
            coarse_fft_pooling, fine_fft_pooling = self.fft(x, 4)  # (seq_len, seg_len, num_hidden)
            coarse_dwt_pooling, fine_dwt_pooling = self.dwt(x, 8)  # (seq_len, seg_len//2, num_hidden)
            print(coarse_dwt_pooling.shape, fine_dwt_pooling.shape, coarse_fft_pooling.shape, fine_fft_pooling.shape)
            return coarse_fft_pooling, fine_fft_pooling, coarse_dwt_pooling, fine_dwt_pooling        

        elif self.scale == [4]:
            coarse_fft_pooling, fine_fft_pooling = self.fft(x, 4)
            print(coarse_fft_pooling.shape, fine_fft_pooling.shape)
            return coarse_fft_pooling, fine_fft_pooling
        
        elif self.scale == [8]:
            coarse_dwt_pooling, fine_dwt_pooling = self.dwt(x, 8)
            print(coarse_dwt_pooling.shape, fine_dwt_pooling.shape)
            return coarse_dwt_pooling, fine_dwt_pooling
    
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
                 num_head, fc_depth=5, orientation='paper', pooling = 'fft'):
        super().__init__()
        self.anchor_scales = anchor_scales
        self.num_scales = len(anchor_scales)
        self.base_model = build_base_model(base_model, num_feature, num_head, orientation)

        self.roi_poolings = [nn.AvgPool1d(scale, stride=1, padding=scale // 2)
                             for scale in anchor_scales]
        if pooling == 'fft':
            self.pooling = Pooling(scale=[4]) 
        elif pooling == 'dwt':
            self.pooling = Pooling(scale=[8])

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
        out = self.base_model(x)
        out = out + x
        out = self.fc1(self.layer_norm(out))
        for fc in self.fc:
            out = fc(out)

        out = out.transpose(2, 1)
        pool_results = [roi_pooling(out) for roi_pooling in self.roi_poolings]
        out = torch.cat(pool_results, dim=0).permute(2, 0, 1)[:-1]

        # out = self.fourier_pooling(out)
        # print(out.shape)

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

        # self.attention_block = nn.ModuleList([self.base_model1 for i in range(attention_depth-1)])

        self.fc_cls = nn.Linear(num_hidden, 1)
        self.fc_loc = nn.Linear(num_hidden, 2)

    def forward(self, x):
        _, seq_len, _ = x.shape
        x = self.base_model2(x + self.base_model1(x)) + x
        # for attention_layer in self.attention_block:
        #     out = attention_layer(x)
        #     x = x + out
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


