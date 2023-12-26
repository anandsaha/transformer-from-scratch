import torch.nn as nn
from torch import Tensor

from model import FeedForwardBlock, MultiHeadAttentionBlock, ResidualConnection


class EncoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super(EncoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection_1 = ResidualConnection(dropout)
        self.residual_connection_2 = ResidualConnection(dropout)

    def forward(self, x: Tensor, src_mask: Tensor = None):
        x = self.residual_connection_1(
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        x = self.residual_connection_2(x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, encoder_layers: nn.ModuleList) -> None:
        super(Encoder, self).__init__()
        self.encoder_layers = encoder_layers

    def forward(self, x: Tensor, src_mask: Tensor = None) -> Tensor:
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, src_mask)
        return x
