import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt
import torch.fft as fft

class Pooling(nn.Module):
    def __init__(self, scale, pooling_type, num_hidden):
        super().__init__()
        self.scale = scale
        self.pooling_type = pooling_type

    def dwt(self, x, scale):
        assert scale == 8
        pooled = []
        coarse_pooling = []
        device = x.device
        for i in range(x.shape[1]):
            end = min(i + scale//2, x.shape[1])
            start = max(0, i - scale//2 + 1)
            segment = x[:, start:end+1, :]
            if segment.shape[1] < scale:
                segment = F.pad(segment, (0, 0, 0, scale - segment.shape[1]))
            coeffs, _ = pywt.dwt(segment.cpu().detach().numpy(), 'db1', axis = 1)
            coeffs = torch.from_numpy(coeffs).to(device)
            pooled.append(coeffs)
            coarse_pooling.append(coeffs.mean(dim=1))
        
        fine_pooling = torch.stack(pooled, dim=1).view(1, x.shape[1],4 * x.shape[2]).to(x.device)
        coarse_pooling = torch.stack(coarse_pooling, dim=1).to(x.device)
        return coarse_pooling, fine_pooling

    def fft(self, x, scale):
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

    def flat_pooling(self, x, scale):
        assert scale * x.shape[-1] == 128*4, "For flat pooling, scale * num_hidden must be 512"
        pooled = []
        for i in range(x.shape[1]):
            end = min(i + scale//2, x.shape[1])
            start = max(0, i - scale//2 + 1)
            segment = x[:, start:end+1, :]
            if segment.shape[1] < scale:
                segment = F.pad(segment, (0, 0, 0, scale - segment.shape[1]))
            pooled.append(segment)

        pooling = torch.stack(pooled, dim=1).view(1, x.shape[1], scale * x.shape[2]).to(x.device) 
        return pooling

    def forward(self, x):
        if self.pooling_type == 'hybrid':
            coarse_fft_pooling, fine_fft_pooling = self.fft(x, 4)  # (seq_len, seg_len, num_hidden)
            coarse_dwt_pooling, fine_dwt_pooling = self.dwt(x, 8)  # (seq_len, seg_len//2, num_hidden)
            # print(coarse_dwt_pooling.shape, fine_dwt_pooling.shape, coarse_fft_pooling.shape, fine_fft_pooling.shape)
            return coarse_fft_pooling, fine_fft_pooling, coarse_dwt_pooling, fine_dwt_pooling        

        elif self.pooling_type == 'fft':
            coarse_fft_pooling, fine_fft_pooling = self.fft(x, 4)
            # print(coarse_fft_pooling.shape, fine_fft_pooling.shape)
            return coarse_fft_pooling, fine_fft_pooling
        
        elif self.pooling_type == 'dwt':
            coarse_dwt_pooling, fine_dwt_pooling = self.dwt(x, 8)
            # print(coarse_dwt_pooling.shape, fine_dwt_pooling.shape)
            return coarse_dwt_pooling, fine_dwt_pooling
        
        elif self.pooling_type == 'flat-pooling':
            pooling = self.flat_pooling(x, 4)
            # print(pooling.shape)
            return pooling
        