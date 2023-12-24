import math
import torch
import torch.nn as nn


class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super(InputEmbeddings, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_length: int, dropout: float) -> None:
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(p=dropout)

        # shape (seq_length, d_model)
        pe = torch.zeros(seq_length, d_model)
        print("PositionalEncoding: Shape of pe is", pe.shape)
        
        # shape (seq_length, 1)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        print("PositionalEncoding: Shape of position is", position.shape)

        # shape (1, d_model / 2)
        denominator = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        print("PositionalEncoding: Shape of denominator is", denominator.shape)

        # Apply sin() to even positions
        pe[:, 0::2] = torch.sin(position * denominator)

        # Apply cos() to odd positions
        pe[:, 1::2] = torch.cos(position * denominator)

        # Add batch dim to pe, shape (1, seq_length, d_model)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10 ** -6) -> None:
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  # alpha is a learnable parameter, multiplied
        self.beta = nn.Parameter(torch.zeros(1))  # beta is a learnable parameter, added


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.beta


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super(FeedForwardBlock, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input dim (batch, seq_length, d_model)
        x = self.linear_2(self.dropout(self.relu(self.linear_1(x))))
        return x
