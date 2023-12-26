import torch.nn as nn
from torch import Tensor

from model import FeedForwardBlock, MultiHeadAttentionBlock, ResidualConnection


class DecoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super(DecoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection_1 = ResidualConnection(dropout)
        self.residual_connection_2 = ResidualConnection(dropout)
        self.residual_connection_3 = ResidualConnection(dropout)

    def forward(
        self,
        x: Tensor,
        encoder_block_output: Tensor,
        src_mask: Tensor,
        target_mask: Tensor,
    ):
        x = self.residual_connection_1(
            x, lambda x: self.self_attention_block(x, x, x, target_mask)
        )
        x = self.residual_connection_2(
            x,
            lambda x: self.cross_attention_block(
                x, encoder_block_output, encoder_block_output, src_mask
            ),
        )
        x = self.residual_connection_3(x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, decoder_block: nn.ModuleList) -> None:
        super(Decoder, self).__init__()
        self.decoder_blocks = decoder_block

    def forward(
        self,
        x: Tensor,
        encoder_block_output: Tensor,
        src_mask: Tensor,
        target_mask: Tensor,
    ) -> Tensor:
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, encoder_block_output, src_mask, target_mask)
        return x


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super(ProjectionLayer, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        # (batch, seq_length, d_model) -> (batch, seq_length, vocab_size)
        return self.linear(x)
