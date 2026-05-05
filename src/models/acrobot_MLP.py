import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(input, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, output)
        )
    def forward(self, x, hx = None):
        return self.seq(x), None