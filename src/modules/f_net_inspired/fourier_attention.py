import torch
import torch.nn as nn

class FFTCalculation(nn.Module):
    def __init__(self):
        super(FFTCalculation, self).__init__()

    def forward(self, features):
        temporal_fft = torch.fft.fft(features, dim=-2).real
        feature_wise_fft = torch.fft.fft(features, dim=-1).real

        # according to paper
        fft_features = torch.fft.fft(torch.fft.fft(features, dim=-1), dim=-2).real
        return fft_features, temporal_fft, feature_wise_fft


class SkipConnection(nn.Module):
    def __init__(self, num_feature):
        super(SkipConnection, self).__init__()
        self.layernorm = nn.LayerNorm(normalized_shape=num_feature)

    def forward(self, x):
        fft_block = FFTCalculation()  
        fft_features, fft_features_temporal, _ = fft_block(x)
        return self.layernorm(fft_features + x), fft_features_temporal


class FNet_layer(nn.Module):
    def __init__(self, num_feature, dropout):
        super(FNet_layer, self).__init__()
        self.layernorm = nn.LayerNorm(num_feature)
        self.fc = nn.Sequential(nn.Linear(num_feature, num_feature),
                                 nn.ReLU(),
                                 nn.Dropout(dropout))

        self.fft_skip = SkipConnection(num_feature=num_feature)

    def forward(self, x): 
        x, fft_temporal_features = self.fft_skip(x)           
        x = self.layernorm(self.fc(x) + x)
        return x

class FNetModel(nn.Module):
    def __init__(self, num_feature, dropout, num_layers):
        super(FNetModel, self).__init__()
        self.dense = nn.Sequential(nn.Linear(num_feature, num_feature),
                                    nn.ReLU(),
                                    nn.Dropout(dropout),
                                    nn.LayerNorm(num_feature))
        self.fnet_layer = FNet_layer(num_feature=num_feature, dropout=dropout)
        self.fft_layer_list = nn.ModuleList([self.fnet_layer for i in range(num_layers)])

    def forward(self, x):
        list_of_temporal_fourier_attention = [] 
        for layer in self.fft_layer_list:
            x, tempral_fft  = layer(x)
            list_of_temporal_fourier_attention.append(tempral_fft)
        return self.dense(x)
    

class FastFNetLayer(nn.Module):
    def __init__(self, num_feature, dropout):
        super(FastFNetLayer, self).__init__()
        self.layernorm = nn.LayerNorm(num_feature)
        self.layernorm_reduced_dim = nn.LayerNorm(num_feature//2)

        self.fft_calculation = FFTCalculation()

        self.dimesion_reduction = nn.Sequential(nn.Linear(num_feature, num_feature//2),
                                    nn.ReLU(),
                                    nn.Dropout(dropout))
        

        self.fc = nn.Sequential(nn.Linear(num_feature//2, num_feature),
                                 nn.ReLU(),
                                 nn.Dropout(dropout))


    def forward(self, x):
        b,seq,num_feature = x.shape
        x_reduced_dimension = self.dimesion_reduction(x)
        x_fft_output, x_fft_output_temporal, _  = self.fft_calculation(x)
        x_fft_output = x_fft_output[:,:,:num_feature//2]
        x = self.layernorm_reduced_dim(x_reduced_dimension + x_fft_output)
        zero_tensor = torch.zeros(b,seq,num_feature, device = x.device)
        zero_tensor[:,:,num_feature//4 : 3*num_feature//4] = x 

        return self.layernorm(zero_tensor + self.fc(x))
    
class FastFNetModel(nn.Module):
    def __init__(self, num_feature, dropout, num_layers):
        super(FastFNetModel, self).__init__()
        self.dense = nn.Sequential(nn.Linear(num_feature, num_feature),
                                    nn.ReLU(),
                                    nn.Dropout(dropout),
                                    nn.LayerNorm(num_feature))
        self.fnet_layer = FastFNetLayer(num_feature=num_feature, dropout=dropout)
        self.fft_layer_list = nn.ModuleList([self.fnet_layer for i in range(num_layers)])

    def forward(self, x):
        list_of_temporal_fourier_attention = [] 
        for layer in self.fft_layer_list:
            x,tempral_fft,_ = layer(x)
            list_of_temporal_fourier_attention.append(tempral_fft)
        return self.dense(x)