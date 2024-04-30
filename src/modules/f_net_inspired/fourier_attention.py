import torch
import torch.nn as nn

class FFTCalculation(nn.Module):
    def __init__(self, num_feature):
        super(FFTCalculation, self).__init__()
        self.num_feature = num_feature

    def forward(self, features):
        temporal_fft = torch.fft.fft(features, dim=-2).real
        feature_wise_fft = torch.fft.fft(features, dim=-1).real

        # according to paper
        fft_features = torch.fft.fft(torch.fft.fft(features, dim=-1), dim=-2).real
        return fft_features, temporal_fft, feature_wise_fft


class SkipConnection(nn.Module):
    def __init__(self, num_feature, ffttype):
        super(SkipConnection, self).__init__()
        self.layernorm = nn.LayerNorm(normalized_shape=num_feature)
        self.ffttype = ffttype

    def forward(self, x):
        fft_block = FFTCalculation(x.size(-1))  # Create FFTCalculation instance based on input size
        if self.ffttype == 'paper':
            fft_features, _, _ = fft_block(x)

        elif self.ffttype == 'temporal':
            _, fft_features, _ = fft_block(x)

        elif self.ffttype == 'feature_wise':
            _, _, fft_features = fft_block(x)

        return self.layernorm(fft_features + x)


class FNet_layer(nn.Module):
    def __init__(self, num_feature, dropout, ffttype, pos_encoding):
        super(FNet_layer, self).__init__()
        self.layernorm = nn.LayerNorm(num_feature)
        self.pos_encoding = pos_encoding
        self.fc = nn.Sequential(nn.Linear(num_feature, num_feature),
                                 nn.ReLU(),
                                 nn.Dropout(dropout))

        self.fft_skip = SkipConnection(num_feature=num_feature, ffttype=ffttype)

    def forward(self, x):
        if self.pos_encoding is not None:
            x = self.fft_skip(x + self.pos_encoding)
            
        x = self.layernorm(self.fc(x) + x)
        return x

class BuildModel(nn.Module):
    def __init__(self, num_feature, dropout, ffttype, pos_encoding, num_layers):
        super(BuildModel, self).__init__()
        self.dense = nn.Sequential(nn.Linear(num_feature, num_feature),
                                    nn.ReLU(),
                                    nn.Dropout(dropout),
                                    nn.LayerNorm(num_feature))
        self.fnet_layer = FNet_layer(num_feature=num_feature, dropout=dropout, ffttype=ffttype, pos_encoding=pos_encoding)
        self.fft_layer_list = nn.ModuleList([self.fnet_layer for i in range(num_layers)])

    def forward(self, x):
        for layer in self.fft_layer_list:
            x = layer(x)
        return self.dense(x)
    

