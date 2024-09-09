import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt
import torch.fft as fft

class Pooling(nn.Module):
    def __init__(self, scale, pooling_type, num_hidden):
        super().__init__()
        self.scale = scale[0]
        self.pooling_type = pooling_type
        if pooling_type == 'dwt':
            self.fc = nn.Sequential(nn.Linear(num_hidden * scale[0]//2, num_hidden), nn.GELU())
        else:
            self.fc = nn.Sequential(nn.Linear(num_hidden * scale[0], num_hidden), nn.GELU())

    def dwt(self, x):
        # assert self.scale == 8
        pooled = []
        coarse_pooling = []
        for i in range(x.shape[1]):
            end = min(i + self.scale // 2, x.shape[1])
            start = max(0, i - self.scale // 2 + 1)
            segment = x[:, start:end + 1, :]
            if segment.shape[1] < self.scale:
                segment = F.pad(segment, (0, 0, 0, self.scale - segment.shape[1]))
            coeffs, _ = pywt.dwt(segment.cpu().detach().numpy(), 'db1', axis=1)
            coeffs = torch.from_numpy(coeffs).to(x.device)
            pooled.append(coeffs)
            coarse_pooling.append(coeffs.mean(dim=1))

        fine_pooling = torch.stack(pooled, dim=1).view(1, x.shape[1], self.scale * x.shape[2]//2).to(x.device)
        coarse_pooling = torch.stack(coarse_pooling, dim=1).to(x.device)
        return coarse_pooling, self.fc(fine_pooling)

    def fft(self, x):
        # assert self.scale == 4
        pooled = []
        coarse_pooling = []
        for i in range(x.shape[1]):
            end = min(i + self.scale // 2, x.shape[1])
            start = max(0, i - self.scale // 2 + 1)
            segment = x[:, start:end + 1, :]
            if segment.shape[1] < self.scale:
                segment = F.pad(segment, (0, 0, 0, self.scale - segment.shape[1]))
            segment_fft = torch.fft.fft(segment, dim=1).real  # (1, 4, num_hidden)
            coarse_pooling.append(segment_fft.mean(dim=1))  # (1, 1, num_hidden)
            pooled.append(segment_fft)

        coarse_pooling = torch.stack(coarse_pooling, dim=1).to(x.device)
        fine_pooling = torch.stack(pooled, dim=1).view(1, x.shape[1], self.scale * x.shape[2]).to(x.device)
        return coarse_pooling, self.fc(fine_pooling)

    def flat_pooling(self, x):
        # assert self.scale * x.shape[-1] == 128 * 4, "For flat pooling, scale * num_hidden must be 512"
        pooled = []
        for i in range(x.shape[1]):
            end = min(i + self.scale // 2, x.shape[1])
            start = max(0, i - self.scale // 2 + 1)
            segment = x[:, start:end + 1, :]
            if segment.shape[1] < self.scale:
                segment = F.pad(segment, (0, 0, 0, self.scale - segment.shape[1]))
            pooled.append(segment)

        pooling = torch.stack(pooled, dim=1).view(1, x.shape[1], self.scale * x.shape[2]).to(x.device)
        return self.fc(pooling)

    def forward(self, x):   

        if self.pooling_type == 'fft':
            coarse_fft_pooling, fine_fft_pooling = self.fft(x)
            return coarse_fft_pooling, fine_fft_pooling

        elif self.pooling_type == 'dwt':
            coarse_dwt_pooling, fine_dwt_pooling = self.dwt(x)
            return coarse_dwt_pooling, fine_dwt_pooling

        elif self.pooling_type == 'flat-pooling':
            pooling = self.flat_pooling(x)
            return pooling
