import torch
import torch.nn as nn

class network(nn.Module):
    def __init__(self, signal_len):
        super(network, self).__init__()
        self.signal_len = signal_len
        self.base_len = signal_len // 2
        self.spatial = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64,self.base_len)
        )

        self.temporal = nn.Sequential(
            # Duplica el tamaño: base_len -> signal_length
            nn.ConvTranspose1d(in_channels=1, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.Conv1d(in_channels=8, out_channels=1, kernel_size=3, padding=1)
        )
    def forward(self, x):
        base = self.spatial(x)          # [Batch, base_len]
        base = base.unsqueeze(1)        # [Batch, 1, base_len]
        wave = self.temporal(base)      # [Batch, 1, signal_length]
        return wave.squeeze(1)          # [Batch, signal_length]