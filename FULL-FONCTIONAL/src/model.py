import torch
import torch.nn as nn
import torch.optim as optim

class LSTMTransformerForex(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim=64, lstm_layers=1, 
                 transformer_dim=64, nhead=4, num_transformer_layers=2, 
                 fc_dim=1, dropout=0.1):
        super(LSTMTransformerForex, self).__init__()
        
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=lstm_hidden_dim,
                            num_layers=lstm_layers,
                            batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0)
        
        if lstm_hidden_dim != transformer_dim:
            self.projection = nn.Linear(lstm_hidden_dim, transformer_dim)
        else:
            self.projection = nn.Identity()

        # Set batch_first=True for Transformer to avoid warnings & no permutation needed
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim,
                                                   nhead=nhead,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        self.fc = nn.Linear(transformer_dim, fc_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, lstm_hidden_dim)
        proj_out = self.projection(lstm_out)  # (batch, seq_len, transformer_dim)

        # Transformer with batch_first=True expects (batch, seq_len, features)
        transformer_out = self.transformer_encoder(proj_out) # (batch, seq_len, transformer_dim)
        
        # Take the last time step
        last_timestep = transformer_out[:, -1, :]  # (batch, transformer_dim)
        out = self.fc(last_timestep)  # (batch, 1)
        return out

def init_model(input_dim, lstm_hidden_dim=64, lstm_layers=1, 
               transformer_dim=64, nhead=4, num_transformer_layers=2, 
               fc_dim=1, dropout=0.1, lr=0.001):
    model = LSTMTransformerForex(input_dim=input_dim,
                                 lstm_hidden_dim=lstm_hidden_dim,
                                 lstm_layers=lstm_layers,
                                 transformer_dim=transformer_dim,
                                 nhead=nhead,
                                 num_transformer_layers=num_transformer_layers,
                                 fc_dim=fc_dim,
                                 dropout=dropout)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, criterion, optimizer
