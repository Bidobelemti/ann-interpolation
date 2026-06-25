import torch
import torch.nn as nn

class SignalNet(nn.Module):
    def __init__(self, signal_len):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),

            nn.Linear(64, 128),
            nn.Tanh(),

            nn.Linear(128, signal_len)
        )

    def forward(self, x):
        return self.net(x)