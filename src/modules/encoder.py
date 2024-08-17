import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.models import build_base_model
import math

class ClassicEncoder(nn.Module):
    def __init__(self, base_model, num_feature, 
                 num_head, attention_depth):
        super().__init__()
        self.base_model = build_base_model(base_model, num_feature, num_head)

        self.layer_norm = nn.LayerNorm(num_feature)

        self.fc1 = nn.Linear(num_feature, num_feature)

        self.attention_block = nn.ModuleList([self.base_model for i in range(attention_depth-1)])

    def forward(self, x):
        _, seq_len, _ = x.shape
        x = self.base_model(x)
        for attention_layer in self.attention_block:
            out = attention_layer(x)
            x = x + out
        out = x

        out = self.fc1(out)
        return out
    

class MultiAttention(nn.Module):
    def __init__(self, num_feature, base_model, orientation, num_segments=5, num_head=8, local_attention_head=4):
        super(MultiAttention, self).__init__()

        self.num_segments = num_segments
        self.global_attention = build_base_model(base_model, num_feature, num_head, orientation)
        self.layer_norm = nn.LayerNorm(num_feature)
        self.fc = nn.Sequential(nn.Linear(num_feature, num_feature),
                                nn.ReLU())

        if self.num_segments is not None:
            assert self.num_segments >= 2, "num_segments must be None or 2+"
            self.local_attention = nn.ModuleList()
            for _ in range(self.num_segments):
                self.local_attention.append(build_base_model(base_model, num_feature, num_head=local_attention_head, orientation=orientation))

    def forward(self, x):
        weighted_value = self.fc(self.global_attention(x))

        if self.num_segments is not None:
            segment_size = math.ceil(x.shape[-2] / self.num_segments)
            for segment in range(self.num_segments):
                left_pos = segment * segment_size
                right_pos = (segment + 1) * segment_size
                local_x = x[:, left_pos:right_pos, :]
                weighted_local_value = self.fc(self.local_attention[segment](local_x))

                normalized_global_value = F.normalize(weighted_value[:, left_pos:right_pos, :], p=2, dim=-1)
                normalized_local_value = F.normalize(weighted_local_value, p=2, dim=-1)

                # Avoid in-place operation
                weighted_value = torch.cat((
                    weighted_value[:, :left_pos, :],
                    normalized_global_value + normalized_local_value,
                    weighted_value[:, right_pos:, :]
                ), dim=1)

        return weighted_value
    

class LocalGlobalEncoder(nn.Module):
    def __init__(self, base_model, orientation, num_feature, num_head, num_segments, local_attention_head):
        super(LocalGlobalEncoder, self).__init__()
        self.multi_attention = MultiAttention(num_feature, base_model, orientation, num_segments, num_head, local_attention_head)
        self.layer_norm = nn.LayerNorm(num_feature)
        self.fc = nn.Sequential(nn.Linear(num_feature, num_feature),
                                nn.ReLU())

    def forward(self, x):
        return self.fc(self.layer_norm(self.multi_attention(x)))
    
    
