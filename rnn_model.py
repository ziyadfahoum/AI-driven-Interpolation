import torch
import torch.nn as nn

class RNNSignalUpsampler(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=14, num_layers=2, upsample_factor=5, dropout=0.05):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # UPGRADE 1: Feature Extraction BEFORE upsampling
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU()
        )
        
        # UPGRADE 2: Overlapping Context Upsampler
        # Kernel size 7, padding 1, stride 5 results in exactly 5x length 
        # but overlaps the edges to prevent "checkerboard" block artifacts.
        self.upsampler = nn.ConvTranspose1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=upsample_factor + 2, 
            stride=upsample_factor,
            padding=1
        )

        gru_dropout = dropout if num_layers > 1 else 0.0
        
        # UPGRADE 3: Bidirectional GRU
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2, # Halved so the bidirectional concat equals hidden_dim
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=True 
        )

        # UPGRADE 4: Deep Output Projection
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, x):
        # x shape: (B, L, 2)
        x = x.permute(0, 2, 1)        # (B, 2, L)
        
        # 1. Extract features from the low-res signal first
        x = self.feature_extractor(x) # (B, hidden_dim, L)
        
        # 2. Upsample with overlapping kernels
        x = self.upsampler(x)         # (B, hidden_dim, L * 5)
        
        x = x.permute(0, 2, 1)        # (B, L * 5, hidden_dim)
        
        # 3. Bidirectional RNN processing
        rnn_out, _ = self.rnn(x)
        
        # UPGRADE 5: Residual / Skip Connection
        # Adds the upsampled features directly to the RNN output to stabilize deep training
        out = rnn_out + x             
        
        # 4. Final non-linear projection
        return self.fc_out(out)