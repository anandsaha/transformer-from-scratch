import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEmbeddings(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model  # e.g. 512
        self.seq_len = seq_len  # e.g. 1024 (context length)
        self.dropout = nn.Dropout(p=dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # Create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # Slight deviation from paper, but good for numerical stability
        denominator = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sin() to even positions
        pe[:, 0::2] = torch.sin(position * denominator)

        # Apply cos() to odd positions
        pe[:, 1::2] = torch.cos(position * denominator)

        # Add batch dim to pe, shape (1, seq_len, d_model)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :].requires_grad_(False))
        return self.dropout(x)

class LayerNormalization(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        self.alpha = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / (var + self.eps)
        return self.alpha * normalized + self.beta

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # W1, b1
        self.dropout = nn.Dropout(p=dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # W2, b2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq_length, d_model) -> (batch, seq_length, d_ff) -> (batch, seq_length, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(p=dropout)
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)  # Wq, bq
        self.w_k = nn.Linear(d_model, d_model)  # Wk, bk
        self.w_v = nn.Linear(d_model, d_model)  # Wv, bv
        self.w_o = nn.Linear(d_model, d_model)  # Wo, bo

    @staticmethod
    def attention(q, k, v, mask, dropout=None):
        d_k = q.shape[-1]
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # (batch, num_heads, seq_length, seq_length)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = torch.softmax(attention_scores, dim=-1)  # (batch, num_heads, seq_length, seq_length)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return torch.matmul(attention_scores, v), attention_scores

    def forward(self, q, k, v, mask=None):
        query = self.w_q(q)  # (batch, seq_length, d_model) -> (batch, seq_length, d_model)
        key = self.w_k(k)  # (batch, seq_length, d_model) -> (batch, seq_length, d_model)
        value = self.w_v(v)  # (batch, seq_length, d_model) -> (batch, seq_length, d_model)
        
        # d_model = num_heads * d_k
        # (batch, seq_length, d_model) -> (batch, seq_length, num_heads, d_k) -> (batch, num_heads, seq_length, d_k)
        query = query.view(query.shape[0], -1, self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], -1, self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], -1, self.num_heads, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (batch, num_heads, seq_length, d_k) -> (batch, seq_length, num_heads, d_k) -> (batch, seq_length, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)
        return self.w_o(x)

class ResidualConnection(nn.Module):

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNormalization(d_model)

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float, d_model: int):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection_1 = ResidualConnection(d_model, dropout)
        self.residual_connection_2 = ResidualConnection(d_model, dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        x = self.residual_connection_1(x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connection_2(x, self.feed_forward_block)
        return x

class Encoder(nn.Module):

    def __init__(self, d_model: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(d_model)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)  # Why this?

class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float, d_model: int):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection_1 = ResidualConnection(d_model, dropout)
        self.residual_connection_2 = ResidualConnection(d_model, dropout)
        self.residual_connection_3 = ResidualConnection(d_model, dropout)

    def forward(self, x: torch.Tensor, encoder_block_output: torch.Tensor, src_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        x = self.residual_connection_1(x, lambda x: self.self_attention_block(x, x, x, target_mask))
        x = self.residual_connection_2(x, lambda x: self.cross_attention_block(x, encoder_block_output, encoder_block_output, src_mask))
        x = self.residual_connection_3(x, self.feed_forward_block)
        return x

class Decoder(nn.Module):

    def __init__(self, d_model: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(d_model)

    def forward(self, x: torch.Tensor, encoder_block_output: torch.Tensor, src_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_block_output, src_mask, target_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq_length, d_model) -> (batch, seq_length, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)
        