import torch
import torch.nn as nn
import math

import torch
from torch import nn, Tensor

# Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, window_size, n_features]``
        """
        return self.pe[:, :x.size(1)]

class TokenEncoding(nn.Module):
    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(n_features, d_model, kernel_size=3, \
            padding=1, padding_mode='circular', bias=False)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, window_size, n_features]``
        """
        x = x.transpose(1, 2).float()
        return self.conv(x).transpose(1, 2)

class Embedding(nn.Module):
    def __init__(self, n_features: int, d_model: int, \
        window_size: int, dropout: float):
        super().__init__()
        self.token_encoding = TokenEncoding(n_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model, window_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, window_size, n_features]``
        """
        x = self.token_encoding(x) + self.positional_encoding(x)
        return self.dropout(x)

class AnomalyAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, \
        window_size: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_size = d_model // n_heads
        assert d_model % n_heads == 0, 'd_model must be divisible by n_heads'
        self.window_size = window_size

        self.Q = nn.Linear(self.d_model, self.d_model)
        self.K = nn.Linear(self.d_model, self.d_model)
        self.V = nn.Linear(self.d_model, self.d_model)
        self.Sigma = nn.Linear(self.d_model, self.n_heads)
        self.linear = nn.Linear(self.d_model, self.d_model)

        self.dropout = nn.Dropout(dropout)

        # anomaly attention
        i = torch.arange(window_size).reshape(-1, 1)
        j = torch.arange(window_size).reshape(1, -1)
        distances = torch.abs(i - j)
        self.register_buffer('distances', distances)

    
    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """
        Arguments:
            q, k, v: Tensor, shape ``[batch_size, window_size, d_model]``
        """
        batch_size = q.size(0)
        # shape: [batch_size, n_heads, window_size]
        sigma = self.Sigma(q).transpose(1, 2)
        # shape: [batch_size, window_size, n_heads, head_size]
        q = self.Q(q).view(batch_size, self.window_size, self.n_heads, self.head_size)
        k = self.K(k).view(batch_size, self.window_size, self.n_heads, self.head_size)
        v = self.V(v).view(batch_size, self.window_size, self.n_heads, self.head_size)

        sigma = torch.pow(3, torch.sigmoid(sigma * 5) + 1e-5) - 1

        # Self-attention
        # shape: [batch_size, n_heads, window_size, window_size]
        scores = torch.einsum("bqhe,bkhe->bhqk", [q, k]) / math.sqrt(self.head_size)
        sigma = sigma.unsqueeze(-1).expand(-1, -1, -1, self.window_size)
        distances = self.distances.unsqueeze(0).unsqueeze(0)
        distances = distances.expand(batch_size, self.n_heads, -1, -1)
        # prior association
        prior = 1. / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-0.5 * torch.pow(distances / sigma, 2))
        # series association
        series = self.dropout(torch.softmax(scores, dim=-1))
        # attention
        # shape: [batch_size, window_size, n_heads, head_size]
        attention = torch.einsum("bhql,blhd->bqhd", [series, v])
        # shape: [batch_size, window_size, d_model]
        attention = attention.reshape(batch_size, self.window_size, self.d_model)
        out = self.linear(attention)
        return out, series, prior

class AnomalyEncoder(nn.Module):
    def __init__(self, window_size: int, d_model: int, n_heads: int, \
        d_ff, dropout: float):
        super().__init__()
        self.attention = AnomalyAttention(d_model, n_heads, window_size, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, window_size, d_model]``
        """
        attention, series, prior = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attention))
        x = self.norm2(x + self.feed_forward(x))
        return x, series, prior




class AnomalyTransformer(nn.Module):
    def __init__(self, n_features: int, window_size: int, \
        d_model: int, n_heads: int, d_ff: int, n_encoder_layers: int, \
        dropout: float):
        super(AnomalyTransformer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = 4 * self.d_model if d_ff is None else d_ff
        self.n_encoder_layers = n_encoder_layers
        self.dropout = dropout

        self.embedding = Embedding(self.n_features, self.d_model, \
            self.window_size, self.dropout)
        
        self.encoders = nn.ModuleList([
            AnomalyEncoder(self.window_size, self.d_model, self.n_heads, \
                self.d_ff, self.dropout) for _ in range(self.n_encoder_layers)
        ])

        self.norm = nn.LayerNorm(self.d_model)

        self.linear = nn.Linear(self.d_model, self.n_features)

    
    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, window_size, n_features]``
        """
        x = self.embedding(x)
        # series and prior list
        series_list, prior_list = [], []
        for encoder in self.encoders:
            x, series, prior = encoder(x)
            series_list.append(series); prior_list.append(prior)
        x = self.norm(x)
        x = self.linear(x)
        return x, series_list, prior_list


def kl_divergence(p, q):
    """
    Arguments:
        p, q: Tensor, shape ``[..., n]``
    """
    return torch.sum(p * torch.log((p+1e-10) / (q+1e-10)), dim=-1)

def symmetrical_kl_divergence(p, q):
    """
    Arguments:
        p, q: Tensor, shape ``[..., n]``
    """
    return kl_divergence(p, q) + kl_divergence(q, p)

