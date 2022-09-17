from copy import deepcopy
import torch
from torch import nn
import math
from torch.nn import functional as f
from torch.nn.init import xavier_uniform_
from torch.nn import Module, ModuleList, Dropout, Linear, LayerNorm


class Embedding(Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.embedding = nn.Embedding(self.cfg.vocab_size, self.cfg.d_model)

    def forward(self, x):
        return self.embedding(x) / math.sqrt(self.cfg.d_model)


class PositionalEncodingLayer(Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        positional_encoding = torch.zeros(self.cfg.pos_enc_max_size, self.cfg.d_model)
        for pos in range(self.cfg.pos_enc_max_size):
            for i in range(0, self.cfg.d_model, 2):
                positional_encoding[pos, i] = math.sin(pos / (10000 ** ((2 * i) / self.cfg.d_model)))
                positional_encoding[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / self.cfg.d_model)))
        self.positional_encoding = positional_encoding.unsqueeze(0)

    def forward(self, x):
        return x + torch.tensor(self.positional_encoding[:, :x.size(1)], requires_grad=False)


class FeedForwardLayer(Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.linear1 = Linear(self.cfg.d_model, self.cfg.dim_feedforward)
        self.linear2 = Linear(self.cfg.dim_feedforward, self.cfg.d_model)
        self.dropout = Dropout(self.cfg.dropout)

    def forward(self, x):
        return self.linear2(self.dropout(f.relu(self.linear1(x))))


class EncoderLayer(Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.self_attention_layer = MultiHeadAttention(self.cfg)
        self.feed_forward_layer = FeedForwardLayer(self.cfg)

        self.norm1 = LayerNorm(self.cfg.d_model, self.cfg.layer_norm_eps)
        self.norm2 = LayerNorm(self.cfg.d_model, self.cfg.layer_norm_eps)

        self.dropout1 = Dropout(self.cfg.dropout)
        self.dropout2 = Dropout(self.cfg.dropout)

    def forward(self, x, input_mask):
        self_attention_input = self.norm1(x)
        x = x + self.dropout1(self.self_attention_layer(self_attention_input,
                                                        self_attention_input,
                                                        self_attention_input,
                                                        input_mask))
        return x + self.dropout2(self.feed_forward_layer(self.norm2(x)))


class DecoderLayer(Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.self_attention_layer = MultiHeadAttention(self.cfg)
        self.encoder_decoder_attention_layer = MultiHeadAttention(self.cfg)
        self.feed_forward_layer = FeedForwardLayer(self.cfg)

        self.norm1 = LayerNorm(self.cfg.d_model, self.cfg.layer_norm_eps)
        self.norm2 = LayerNorm(self.cfg.d_model, self.cfg.layer_norm_eps)
        self.norm3 = LayerNorm(self.cfg.d_model, self.cfg.layer_norm_eps)

        self.dropout1 = Dropout(self.cfg.dropout)
        self.dropout2 = Dropout(self.cfg.dropout)
        self.dropout3 = Dropout(self.cfg.dropout)

    def forward(self, x, encoder_out, input_mask, target_mask):
        self_attention_input = self.norm1(x)
        x = x + self.dropout1(
            self.self_attention_layer(self_attention_input, self_attention_input, self_attention_input,
                                      target_mask))
        x = x + self.dropout2(self.encoder_decoder_attention_layer(self.norm2(x), encoder_out, encoder_out, input_mask))
        return x + self.dropout3(self.feed_forward_layer(self.norm3(x)))


class Encoder(Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.embedding_layer = Embedding(self.cfg)
        self.pos_encoding_layer = PositionalEncodingLayer(self.cfg)
        self.encoder_layers = ModuleList([deepcopy(EncoderLayer(self.cfg)) for _ in range(self.cfg.num_encoder_layers)])
        self.layer_norm = LayerNorm(self.cfg.d_model, self.cfg.layer_norm_eps)

    def forward(self, input_, input_mask):
        x = self.embedding_layer(input_)
        x = self.pos_encoding_layer(x)
        for enc_layer in self.encoder_layers:
            x = enc_layer(x, input_mask)
        return self.layer_norm(x)


class Decoder(Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.embedding_layer = Embedding(self.cfg)
        self.pos_encoding_layer = PositionalEncodingLayer(self.cfg)
        self.decoder_layers = ModuleList([deepcopy(DecoderLayer(self.cfg))
                                          for _ in range(self.cfg.num_decoder_layers)])
        self.layer_norm = LayerNorm(self.cfg.d_model, self.cfg.layer_norm_eps)

    def forward(self, target, encoder_out, input_mask, target_mask):
        x = self.embedding_layer(target)
        x = self.pos_encoding_layer(x)
        for dec_layer in self.decoder_layers:
            x = dec_layer(x, encoder_out, input_mask, target_mask)
        return self.layer_norm(x)


class MultiHeadAttention(Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.d_k = self.cfg.d_model // self.cfg.h
        self.Q_projection_layer = Linear(self.cfg.d_model, self.cfg.d_model)
        self.V_projection_layer = Linear(self.cfg.d_model, self.cfg.d_model)
        self.K_projection_layer = Linear(self.cfg.d_model, self.cfg.d_model)
        self.dropout = Dropout(self.cfg.dropout)
        self.last_projection_layer = Linear(self.cfg.d_model, self.cfg.d_model)

    @staticmethod
    def attention(q, k, v, d_k, mask=None, dropout=None):
        q_k_mm = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            q_k_mm = q_k_mm.masked_fill(mask == 0, -1e9)
        out = f.softmax(q_k_mm, -1)

        if dropout is not None:
            out = dropout(out)
        return torch.matmul(out, v)

    def forward(self, q, k, v, mask=None):
        b_size = q.size(0)
        k = self.K_projection_layer(k).view(b_size, -1, self.cfg.h, self.d_k).transpose(1, 2)
        q = self.Q_projection_layer(q).view(b_size, -1, self.cfg.h, self.d_k).transpose(1, 2)
        v = self.V_projection_layer(v).view(b_size, -1, self.cfg.h, self.d_k).transpose(1, 2)

        attention_map = self.attention(q, k, v, self.d_k, mask, self.dropout
                                       ).transpose(1, 2).contiguous().view(b_size, -1, self.cfg.d_model)
        return self.last_projection_layer(attention_map)


class Transformer(Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.encoder = Encoder(self.cfg)
        self.decoder = Decoder(self.cfg)
        self.linear = Linear(self.cfg.d_model, self.cfg.vocab_size)

        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, input_, target, input_mask, target_mask):
        encoder_output = self.encoder(input_, input_mask)
        decoder_output = self.decoder(target, encoder_output, input_mask, target_mask)
        out = self.linear(decoder_output)
        return out


def get_model(cfg):
    model = Transformer(cfg)
    return model
