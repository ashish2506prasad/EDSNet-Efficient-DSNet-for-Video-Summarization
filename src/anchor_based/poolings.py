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
        self.fc_list = nn.ModuleList([])
        for scale in self.scale:
            self.fc_list.append(nn.Linear(scale, 4, bias=False))

    def dwt(self, x):
        poolings_list = []
        for i, scale in enumerate(self.scale):
            segments_list = []
            for j in range(x.shape[1]):
                end = min(j + scale // 2, x.shape[1])
                start = max(0, j - scale // 2 + 1)
                segment = x[:, start:end+1, :]
                if segment.shape[1] < scale:
                    segment = F.pad(segment, (0, 0, 0, scale - segment.shape[1]))
                
                coeffs, _ = pywt.dwt(segment.cpu().detach().numpy(), 'db1', axis=1)
                coeffs = torch.from_numpy(coeffs).to(x.device)
                segments_list.append(coeffs)

            segment_tensor = torch.cat(segments_list, dim=0).permute(0, 2, 1).to(x.device)
            segment_tensor = self.fc_list[i-1](segment_tensor)  # Correct index here
            segment_tensor = segment_tensor.permute(0, 2, 1)

            poolings_list.append(segment_tensor)

        return poolings_list  # Only one list is returned

    def flat_pooling(self, x):
        poolings_list = []
        for i, scale in enumerate(self.scale):
            segments_list = []
            for j in range(x.shape[1]):  # Corrected loop variable to `j`
                end = min(j + scale // 2, x.shape[1])
                start = max(0, j - scale // 2 + 1)
                segment = x[:, start:end+1, :]
                if segment.shape[1] < scale:
                    segment = F.pad(segment, (0, 0, 0, scale - segment.shape[1]))

                segments_list.append(segment)
    
            segment_tensor = torch.cat(segments_list, dim=0).permute(0, 2, 1).to(x.device)  # (seq_len, num_hidden, scale)
            segment_tensor = self.fc_list[i](segment_tensor)  # Correct index
            segment_tensor = segment_tensor.permute(0, 2, 1)  # (seq_len, scale, num_hidden)

            poolings_list.append(segment_tensor)
        return poolings_list
    
    def fft(self, x):
        poolings_list = []
        for i, scale in enumerate(self.scale):
            segments_list = []
            for j in range(x.shape[1]):
                end = min(j + scale // 2, x.shape[1])
                start = max(0, j - scale // 2 + 1)
                segment = x[:, start:end+1, :]
                if segment.shape[1] < scale:
                    segment = F.pad(segment, (0, 0, 0, scale - segment.shape[1]))

                segments_list.append(segment)

            segment_tensor = torch.cat(segments_list, dim=0).permute(0, 2, 1).to(x.device)  # (seq_len, num_hidden, scale)
            segment_tensor = self.fc_list[i-1](segment_tensor)
            segment_tensor = fft.fft(segment_tensor, dim=-1).real
            segment_tensor = segment_tensor.permute(0, 2, 1)

            poolings_list.append(segment_tensor)

        return poolings_list


    def forward(self, x):
        if self.pooling_type == 'fft':
            pooling_list = self.fft(x)
            return pooling_list
        
        elif self.pooling_type == 'dwt':
            coarse_dwt_pooling = self.dwt(x)
            return coarse_dwt_pooling  # Only one list returned
        
        elif self.pooling_type == 'flat-pooling':
            pooling_list = self.flat_pooling(x)
            return pooling_list