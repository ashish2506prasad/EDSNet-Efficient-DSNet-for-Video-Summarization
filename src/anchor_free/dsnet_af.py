from torch import nn

from anchor_free import anchor_free_helper
from modules.models import build_base_model


class DSNetAF(nn.Module):
    def __init__(self, base_model, num_feature, num_hidden, num_head, fc_depth):
        super().__init__()
        self.base_model = build_base_model(base_model, num_feature, num_head)
        self.layer_norm = nn.LayerNorm(num_feature)

        self.fc1 = nn.Linear(num_feature, num_hidden)
        self.fc_block = nn.Sequential(nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LayerNorm(num_hidden)
            )
        self.fc = nn.ModuleList([self.fc_block for i in range(fc_depth)])
        
        self.fc_cls = nn.Linear(num_hidden, 1)
        self.fc_loc = nn.Linear(num_hidden, 2)
        self.fc_ctr = nn.Linear(num_hidden, 1)

    def forward(self, x):
        _, seq_len, _ = x.shape
        out = self.base_model(x)

        out = out + x
        out = self.layer_norm(out)

        out = self.fc1(out)
        for layer in self.fc:
            out = layer(out)

        pred_cls = self.fc_cls(out).sigmoid().view(seq_len)
        pred_loc = self.fc_loc(out).exp().view(seq_len, 2)

        pred_ctr = self.fc_ctr(out).sigmoid().view(seq_len)

        return pred_cls, pred_loc, pred_ctr

    def predict(self, seq):
        pred_cls, pred_loc, pred_ctr = self(seq)

        pred_cls *= pred_ctr
        pred_cls /= pred_cls.max() + 1e-8

        pred_cls = pred_cls.cpu().numpy()
        pred_loc = pred_loc.cpu().numpy()

        pred_bboxes = anchor_free_helper.offset2bbox(pred_loc)
        return pred_cls, pred_bboxes
    

class DSNetAF_DeepAttention(nn.Module):
    def __init__(self, base_model, num_feature, num_hidden, num_head, fc_depth):
        super().__init__()
        self.base_model1 = build_base_model(base_model, num_feature, num_head//2)
        self.fc = nn.Linear(num_feature, num_feature)
        self.base_model2 = build_base_model(base_model, num_feature, num_head)

        self.layer_norm = nn.LayerNorm(num_feature)

        self.fc1 = nn.Linear(num_feature, num_hidden)
        self.fc_block = nn.Sequential(nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LayerNorm(num_hidden)
            )
        self.fc = nn.ModuleList([self.fc_block for i in range(fc_depth)])
        
        self.fc_cls = nn.Linear(num_hidden, 1)
        self.fc_loc = nn.Linear(num_hidden, 2)
        self.fc_ctr = nn.Linear(num_hidden, 1)

    def forward(self, x):
        _, seq_len, _ = x.shape
        out1 = self.base_model1(x)
        out = self.base_model2(out1)
        out = x + out1 + out
        out = self.layer_norm(out)

        out = self.fc1(out)
        for layer in self.fc:
            out = layer(out)

        pred_cls = self.fc_cls(out).sigmoid().view(seq_len)
        pred_loc = self.fc_loc(out).exp().view(seq_len, 2)

        pred_ctr = self.fc_ctr(out).sigmoid().view(seq_len)

        return pred_cls, pred_loc, pred_ctr

    def predict(self, seq):
        pred_cls, pred_loc, pred_ctr = self(seq)

        pred_cls *= pred_ctr
        pred_cls /= pred_cls.max() + 1e-8

        pred_cls = pred_cls.cpu().numpy()
        pred_loc = pred_loc.cpu().numpy()

        pred_bboxes = anchor_free_helper.offset2bbox(pred_loc)
        return pred_cls, pred_bboxes
