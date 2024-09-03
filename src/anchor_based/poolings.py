import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt
import torch.fft as fft

class Pooling(nn.Module):
    def __init__(self, scale: list, pooling_type: str):
        super().__init__()
        self.scale = sorted(scale)
        self.pooling_type = pooling_type
        self.fc_list = nn.ModuleList([
            nn.Linear(s if pooling_type != 'dwt' else s // 2, 4, bias=False)
            for s in self.scale
        ])

    def segment_and_pad(self, x, scale):
        segments_list = []
        for j in range(x.shape[1]):
            start, end = max(0, j - scale // 2 + 1), min(j + scale // 2, x.shape[1])
            segment = x[:, start:end+1, :]
            if segment.shape[1] < scale:
                segment = F.pad(segment, (0, 0, 0, scale - segment.shape[1]))
            segments_list.append(segment)
        return segments_list

    def dwt(self, x):
        assert self.scale[0] == 8
        poolings_list = []
        for i, scale in enumerate(self.scale):
            segments_list = self.segment_and_pad(x, scale)
            coeffs_list = [
                torch.from_numpy(pywt.dwt(segment.cpu().numpy(), 'db1', axis=1)[0]).to(x.device)
                for segment in segments_list
            ]
            segment_tensor = torch.cat(coeffs_list, dim=0).permute(0, 2, 1).to(x.device)
            segment_tensor = self.fc_list[i](segment_tensor).permute(0, 2, 1)
            poolings_list.append(segment_tensor)
        return poolings_list

    def flat_pooling(self, x):
        assert self.scale[0] == 4
        poolings_list = []
        for i, scale in enumerate(self.scale):
            segments_list = self.segment_and_pad(x, scale)
            normalized_segments = [F.normalize(segment, dim=-1) for segment in segments_list]
            segment_tensor = torch.cat(normalized_segments, dim=0).permute(0, 2, 1).to(x.device)
            segment_tensor = self.fc_list[i](segment_tensor).permute(0, 2, 1)
            poolings_list.append(segment_tensor)
        return poolings_list

    def fft(self, x):
        assert self.scale[0] == 4
        poolings_list = []
        for i, scale in enumerate(self.scale):
            segments_list = self.segment_and_pad(x, scale)
            fft_segments = [
                F.normalize(fft.fft(segment.permute(0, 2, 1), dim=-1).real, dim=-1)
                for segment in segments_list
            ]
            segment_tensor = torch.cat(fft_segments, dim=0).to(x.device)
            segment_tensor = segment_tensor.permute(0, 2, 1)
            poolings_list.append(segment_tensor)
        return poolings_list

    def forward(self, x):
        if self.pooling_type == 'fft':
            return self.fft(x)
        elif self.pooling_type == 'dwt':
            return self.dwt(x)
        elif self.pooling_type == 'flat-pooling':
            return self.flat_pooling(x)
