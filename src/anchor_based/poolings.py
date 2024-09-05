import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt
import torch.fft as fft

# class Pooling(nn.Module):
#     def __init__(self, scale: list, pooling_type: str):
#         super().__init__()
#         self.scale = sorted(scale)
#         self.pooling_type = pooling_type
#         self.fc_list = nn.ModuleList([
#             nn.Linear(s if pooling_type != 'dwt' else s // 2, 4, bias=False)
#             for s in self.scale
#         ])

#     def segment_and_pad(self, x, scale):
#         segments_list = []
#         for j in range(x.shape[1]):
#             start, end = max(0, j - scale // 2 + 1), min(j + scale // 2, x.shape[1])
#             segment = x[:, start:end+1, :]
#             if segment.shape[1] < scale:
#                 segment = F.pad(segment, (0, 0, 0, scale - segment.shape[1]))
#             segments_list.append(segment)
#         return segments_list

#     def dwt(self, x):
#         assert self.scale[0] == 8
#         poolings_list = []
#         for i, scale in enumerate(self.scale):
#             segments_list = self.segment_and_pad(x, scale)
#             coeffs_list = [
#                 torch.from_numpy(pywt.dwt(segment.cpu().numpy(), 'db1', axis=1)[0]).to(x.device)
#                 for segment in segments_list
#             ]
#             segment_tensor = torch.cat(coeffs_list, dim=0).permute(0, 2, 1).to(x.device)
#             segment_tensor = self.fc_list[i](segment_tensor).permute(0, 2, 1)
#             poolings_list.append(segment_tensor)
#         return poolings_list

#     def flat_pooling(self, x):
#         assert self.scale[0] == 4
#         poolings_list = []
#         for i, scale in enumerate(self.scale):
#             segments_list = self.segment_and_pad(x, scale)
#             segment_tensor = torch.cat(segments_list, dim=0).permute(0, 2, 1).to(x.device)
#             segment_tensor = self.fc_list[i](segment_tensor).permute(0, 2, 1)
#             poolings_list.append(segment_tensor)
#         return poolings_list

#     def fft(self, x):
#         assert self.scale[0] == 4
#         poolings_list = []
#         for i, scale in enumerate(self.scale):
#             segments_list = self.segment_and_pad(x, scale)
#             fft_segments = [ fft.fft(segment, dim=1).real   # [(1, scale, num_hidden), (1, scale, num_hidden)]
#                     for segment in segments_list
#             ]
#             segment_tensor = torch.cat(fft_segments, dim=0).to(x.device)
#             segment_tensor = segment_tensor
#             poolings_list.append(segment_tensor)
#         return poolings_list

#     def forward(self, x):
#         if self.pooling_type == 'fft':
#             return self.fft(x)
#         elif self.pooling_type == 'dwt':
#             return self.dwt(x)
#         elif self.pooling_type == 'flat-pooling':
#             return self.flat_pooling(x)


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
        assert self.scale == 8
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

        fine_pooling = torch.stack(pooled, dim=1).view(1, x.shape[1], 4 * x.shape[2]).to(x.device)
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
