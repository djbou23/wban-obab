"""Deep-learning baseline architectures for the tabular flow-feature benchmark.

All models treat each sample as a length-1 'sequence' of feature vectors reshaped to
(batch, seq_len=1, features) so the same interface works whether we later window
multiple flows into a real sequence or keep the current one-row-per-sample setup.
Kept intentionally simple/standard since these are baselines, not the paper's novelty.
"""
import torch
import torch.nn as nn


class CNNBaseline(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):  # x: (batch, n_features)
        x = x.unsqueeze(1)  # (batch, 1, n_features)
        x = self.net(x).squeeze(-1)
        return self.fc(x)


class LSTMBaseline(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x):  # treat each feature as one timestep
        x = x.unsqueeze(-1)  # (batch, n_features, 1)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


class BiGRUAttention(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden=64):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden * 2, 1)
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)  # (batch, n_features, 1)
        out, _ = self.gru(x)  # (batch, n_features, hidden*2)
        weights = torch.softmax(self.attn(out), dim=1)  # (batch, n_features, 1)
        context = (out * weights).sum(dim=1)  # (batch, hidden*2)
        return self.fc(context)


class TransformerBaseline(nn.Module):
    def __init__(self, n_features: int, n_classes: int, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_features, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                     dim_feedforward=d_model * 2, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)  # (batch, n_features, 1)
        x = self.input_proj(x) + self.pos_embed
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)


def build_model(name: str, n_features: int, n_classes: int) -> nn.Module:
    registry = {
        "CNN": CNNBaseline,
        "LSTM": LSTMBaseline,
        "BiGRU_Attention": BiGRUAttention,
        "Transformer": TransformerBaseline,
    }
    return registry[name](n_features, n_classes)
