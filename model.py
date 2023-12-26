import math

import torch
import torch.nn as nn
from torch import Tensor


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super(InputEmbeddings, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq_length) -> (batch, seq_length, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_length: int, dropout: float) -> None:
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(p=dropout)

        # shape (seq_length, d_model)
        pe = torch.zeros(seq_length, d_model)

        # shape (seq_length, 1)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)

        # shape (1, d_model / 2)
        denominator = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sin() to even positions
        pe[:, 0::2] = torch.sin(position * denominator)

        # Apply cos() to odd positions
        pe[:, 1::2] = torch.cos(position * denominator)

        # Add batch dim to pe, shape (1, seq_length, d_model)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq_length, d_model) -> (batch, seq_length, d_model)
        x = x + self.pe[:, : x.shape[1], :].requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(
            torch.ones(1)
        )  # alpha is a learnable parameter, multiplied
        self.beta = nn.Parameter(torch.zeros(1))  # beta is a learnable parameter, added

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #  (batch, seq_length, d_model) -> (batch, seq_length, d_model)
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


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super(MultiHeadAttentionBlock, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(p=dropout)
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

    @staticmethod
    def attention(
        q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None, dropout: nn.Dropout = None
    ) -> Tensor:
        d_k = q.shape[-1]

        # (batch, num_heads, seq_length, d_k) -> (batch, num_heads, seq_length, seq_length)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_scores = torch.softmax(
            attention_scores, dim=-1
        )  # (batch, num_heads, seq_length, seq_length)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return torch.matmul(attention_scores, v), attention_scores

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask=None) -> torch.Tensor:
        query_prime = self.w_q(
            q
        )  # (batch, seq_length, d_model) -> (batch, seq_length, d_model)
        key_prime = self.w_k(k)
        value_prime = self.w_v(v)

        shape = query_prime.shape

        # (batch, seq_length, d_model) -> (batch, num_heads, seq_length, d_k)
        query = query_prime.view(
            shape[0], shape[1], self.num_heads, self.d_k
        ).transpose(1, 2)
        key = key_prime.view(shape[0], shape[1], self.num_heads, self.d_k).transpose(
            1, 2
        )
        value = value_prime.view(
            shape[0], shape[1], self.num_heads, self.d_k
        ).transpose(1, 2)

        x, attention_scores = self.attention(query, key, value, mask, self.dropout)

        # (batch, num_heads, seq_length, d_k) -> (batch, seq_length, num_heads, d_k) -> (batch, seq_length, d_model)
        x = x.transpose(1, 2).contiguous().view(shape[0], -1, self.d_model)

        # (batch, seq_length, d_model) -> (batch, seq_length, d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float = 0.1) -> None:
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = LayerNormalization()

    def forward(self, x: Tensor, sublayer: nn.Module) -> Tensor:
        # (batch, seq_length, d_model) -> (batch, seq_length, d_model)
        return x + self.dropout(sublayer(self.layer_norm(x)))
