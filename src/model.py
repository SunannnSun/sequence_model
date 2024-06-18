# src/model.py

import torch
import torch.nn as nn
import math



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerRegressionModel(nn.Module):
    def __init__(self, input_dim, nhead, num_encoder_layers, dim_feedforward, output_dim, dropout=0.1):
        super(TransformerRegressionModel, self).__init__()

        self.input_linear = nn.Linear(input_dim, dim_feedforward)
        self.positional_encoding = PositionalEncoding(d_model=dim_feedforward)

        encoder_layers = nn.TransformerEncoderLayer(d_model=dim_feedforward, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(dim_feedforward, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, output_dim)

    def forward(self, src):
        src = self.input_linear(src)  
        src = self.positional_encoding(src) 
        
        src = src.permute(1, 0, 2)  # Transformer expects [seq_length, batch_size, input_dim]
        output = self.transformer_encoder(src)
        output = output.permute(1, 0, 2)  

        output = self.dropout(output[:, -1, :])  
        output = torch.relu(self.fc1(output))  
        output = self.fc2(output) 

        return output