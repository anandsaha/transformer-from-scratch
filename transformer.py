from torch import Tensor, nn
import model


class Transformer(nn.Module):
    def __init__(
        self,
        the_encoder: model.Encoder,
        the_decoder: model.Decoder,
        projection: model.ProjectionLayer,
        src_embed: model.InputEmbeddings,
        tgt_embed: model.InputEmbeddings,
        src_pos: model.PositionalEncoding,
        tgt_pos: model.PositionalEncoding,
        projection_layer: model.ProjectionLayer,
    ) -> None:
        
        super(Transformer, self).__init__()

        self.encoder = the_encoder
        self.decoder = the_decoder
        self.projection = projection
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src: Tensor, src_mask: Tensor) -> Tensor:
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(
        self, encoder_output: Tensor, src_mask: Tensor, tgt: Tensor, tgt_mask: Tensor
    ) -> Tensor:
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    num_heads: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048,
) -> Transformer:
    # Create the embedding layers
    src_embed = model.InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = model.InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = model.PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = model.PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = model.MultiHeadAttentionBlock(
            d_model, num_heads, dropout
        )
        feed_forward_block = model.FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = model.EncoderBlock(
            encoder_self_attention_block, feed_forward_block, dropout
        )
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = model.MultiHeadAttentionBlock(
            d_model, num_heads, dropout
        )
        decoder_cross_attention_block = model.MultiHeadAttentionBlock(
            d_model, num_heads, dropout
        )
        feed_forward_block = model.FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = model.DecoderBlock(
            decoder_self_attention_block,
            decoder_cross_attention_block,
            feed_forward_block,
            dropout,
        )
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    the_encoder = model.Encoder(nn.ModuleList(encoder_blocks))
    the_decoder = model.Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = model.ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(the_encoder,
                              the_decoder,
                              projection_layer,
                              src_embed,
                              tgt_embed,
                              src_pos,
                              tgt_pos,
                              projection_layer)
