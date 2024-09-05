import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt

class dwt_calculation(nn.Module):
    def __init__(self, wavelet='haar'):
        super(dwt_calculation, self).__init__()
        self.wavelet = wavelet

    def forward(self, x):
        cA, cD = pywt.dwt(x.cpu().detach().numpy(), self.wavelet, axis=1)
        return cA, cD
    
class DwtNet(nn.Module):
    def __init__(self, num_feature, wavelet='haar', dropout=0.5):
        super(DwtNet, self).__init__()
        self.dwt_block = dwt_calculation(wavelet=wavelet)
        self.fc = nn.Sequential(nn.Linear(num_feature, num_feature),
                                nn.GELU(),
                                nn.Dropout(0.5))
        self.layernorm = nn.LayerNorm(num_feature)
        self.num_feature = num_feature
        self.transconv = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=2, stride=2, padding=0)


    def forward(self, x):
        seq_len = x.shape[1]
        cA, cD = self.dwt_block(x)
        x = self.fc(torch.from_numpy(cA).to(x.device))
        x = self.layernorm(x + torch.from_numpy(cD).to(x.device))
        x = self.transconv(x.permute(2, 0, 1)) # (batch, (seq_len//2)*2, num_feature)
        x = x.permute(1, 2, 0)
        if x.shape[1] != seq_len:
            x = x[:, :seq_len, :]   

        x = x.view(-1, seq_len, self.num_feature)
        return self.fc(x)