import torch
import torch.nn as nn

class FFTCalculation(nn.Module):
    def __init__(self, orientation):
        super(FFTCalculation, self).__init__()
        self.orientation = orientation

    def forward(self, features):
        if self.orientation == 'temporal':
            temporal_fft = torch.fft.fft(features, dim=-2).real
            return temporal_fft
        elif self.orientation == 'feature_wise':
            feature_wise_fft = torch.fft.fft(features, dim=-1).real
            return feature_wise_fft
        elif self.orientation == 'paper':
            fft_features_paper = torch.fft.fft(torch.fft.fft(features, dim=-1), dim=-2).real
            return fft_features_paper


class SkipConnection(nn.Module):
    def __init__(self, num_feature, orientation):
        super(SkipConnection, self).__init__()
        self.layernorm = nn.LayerNorm(normalized_shape=num_feature)
        self.orientation = orientation

    def forward(self, x):
        fft_block = FFTCalculation(orientation=self.orientation)  
        fft_features = fft_block(x)
        return self.layernorm(fft_features + x)


class FNet_layer(nn.Module):
    def __init__(self, num_feature, dropout, orientation):
        super(FNet_layer, self).__init__()
        self.layernorm = nn.LayerNorm(num_feature)
        self.fc = nn.Sequential(nn.Linear(num_feature, num_feature),
                                 nn.ReLU(),
                                 nn.Dropout(dropout))

        self.fft_skip = SkipConnection(num_feature=num_feature, orientation=orientation)

    def forward(self, x): 
        x = self.fft_skip(x)           
        x = self.layernorm(self.fc(x) + x)
        return x

class FNetModel(nn.Module):
    def __init__(self, num_feature, dropout, num_layers, orientation):
        super(FNetModel, self).__init__()
        self.orientation = orientation
        self.dense = nn.Sequential(nn.Linear(num_feature, num_feature),
                                    nn.ReLU(),
                                    nn.Dropout(dropout),
                                    nn.LayerNorm(num_feature))
        self.fnet_layer = FNet_layer(num_feature=num_feature, dropout=dropout, orientation=orientation)
        self.fft_layer_list = nn.ModuleList([self.fnet_layer for i in range(num_layers)])

    def forward(self, x): 
        for layer in self.fft_layer_list:
            x  = layer(x)
        return self.dense(x)
