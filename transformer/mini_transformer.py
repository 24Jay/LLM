import torch
import math


class InputEmbedding(torch.nn.Module):

    def __init__(self, d_model: int, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = torch.nn.Embedding(self.vocab_size, self.d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, seq_len: int, d_model: int, dropout: float):
        super().__init__()

        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = torch.nn.Dropout(dropout)

        pe = torch.zeros(self.seq_len, self.d_model)

        # create position vector : (seq_len, 1)
        positions = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.pow(10000, torch.arange(0, d_model, 2).float() / d_model)

        # 偶数用sin, 奇数用cos
        pe[:, 0::2] = torch.sin(positions / div_term)
        pe[:, 1::2] = torch.cos(positions / div_term)

        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + (self.pe[:, : self.x.shape[1], :]).requires_grad_(False)
        return x


pe = PositionalEmbedding(seq_len=50, d_model=512, dropout=0.1)

pe.pe.numpy()

import matplotlib.pyplot as plt


map = pe.pe.numpy().squeeze(0)
print(map)
plt.imshow(map)

plt.show()


class LayerNormalization(torch.nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

        self.alpha = torch.nn.Parameter(torch.ones(1))  # x
        self.bias = torch.nn.Parameter(torch.zeros(1))  # +

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / math.sqrt(std + self.eps) + self.bias


class FeedForwardBlock(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = torch.nn.Linear(d_model, d_ff, bias=True)  # w1 b1
        self.dropout = torch.nn.Dropout(dropout)

        self.linear_2 = torch.nn.Linear(d_ff, d_model, bias=True)  # w2 b2
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)

        return : (batch, seq_len, d_ff  )
        """
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttention(torch.nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()

        assert d_model % h == 0, "d_model must be divided by h"
        self.d_model = d_model
        self.h = h
        self.d_k = d_model / h
        self.dropout = torch.nn.Dropout(dropout)

        self.wq = torch.nn.Linear(d_model, d_model)
        self.wk = torch.nn.Linear(d_model, d_model)
        self.wv = torch.nn.Linear(d_model, d_model)
        self.wo = torch.nn.Linear(d_model, d_model)

    @staticmethod
    def self_attention(query, key, value, mask, dropout: torch.nn.Dropout):
        d_k = query.shape[-1]

        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask:
            attention_score.masked_fill_(mask == 0, -1e9)
        attention_score = attention_score.softmax(
            dim=-1
        )  # (batch, head, seq_len, seq_len)

        if dropout:
            attention_score = dropout(attention_score)

        return (attention_score @ value), attention_score

    def forward(self, q, k, v, mask):
        query = self.wq(q)  # batch, seq_len, d_model -> batch, seq_len, d_model
        key = self.wk(k)  # batch, seq_len, d_model -> batch, seq_len, d_model
        value = self.wv(v)  # batch, seq_len, d_model -> batch, seq_len, d_model

        # transpose : (batch, seq_len, d_model) -> (batch, seq_len, h, d_k)-> (batch, head, seq_len, d_k = emb_len)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(
            1, 2
        )
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(
            1, 2
        )

        x, attention_score = MultiHeadAttention.self_attention(
            query, key, value, mask, self.dropout
        )

        # batch, head, seq_len, d_k -> batch, head, d_k, seq_len -> batch, seq_len, d_model
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.wo(x)


class ResidualConnection(torch.nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()

        self.dropout = torch.nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        """
        paper: sublayer -> norm

        this implementation: norm -> sublayer
        """
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(torch.nn.Module):
    def __init__(
        self,
        self_attention: MultiHeadAttention,
        feed_forward: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()

        self.attention = self_attention
        self.feed_forward = feed_forward

        self.residual_connection = torch.nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(2)]
        )

    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.attention(x, x, x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward)


class Encoder(torch.nn.Module):
    def __init__(self, layers: torch.nn.ModuleList) -> None:
        super().__init__()

        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(torch.nn.Module):
    def __init__(
        self,
        self_attention: MultiHeadAttention,
        cross_attention: MultiHeadAttention,
        feedforward: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()

        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feedforward = feedforward

        self.dropout = torch.nn.Dropout(dropout)
        self.norm = LayerNormalization()

        self.residual_connection = torch.nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(3)]
        )

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection[0](
            x, lambda x: self.self_attention(x, x, x, tgt_mask)
        )
        x = self.residual_connection[1](
            x,
            lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask),
        )
        x = self.residual_connection[2](x, lambda x: self.feedforward(x))
        return x


class ProjectionLayer(torch.nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        pass
