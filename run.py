import torch
import model

d_model = 512
d_seq_len = 1000
d_batche_size = 10
num_heads = 4

test_tensor = torch.rand(d_batche_size, d_seq_len, d_model)


print("Testing PositionalEncoding", "="*20)
p = model.PositionalEncoding(d_model, d_seq_len, 0.1)

print("Testing LayerNormalization", "="*20)
l = model.LayerNormalization()
l.forward(test_tensor)

print("Testing FeedForwardBlock", "="*20)
ff = model.FeedForwardBlock(d_model, 2048, 0.1)
ff.forward(test_tensor)

print("Testing Encoder", "="*20)
el = model.EncoderLayer(model.MultiHeadAttentionBlock(d_model, num_heads),
                        model.FeedForwardBlock(d_model, d_model // num_heads),
                        0.1)
e = model.Encoder(el, 10)
e.forward(test_tensor)
