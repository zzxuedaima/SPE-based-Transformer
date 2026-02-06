from torch import Tensor
import torch.nn.functional as f
import numpy as np
import torch
from torch import nn

def feed_forward(dim_input: int = 16, dim_feedforward: int = 32) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedforward),
        nn.ReLU(),
        nn.Linear(dim_feedforward, dim_input),
    )

class Residual(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: Tensor) -> Tensor:
        out = self.sublayer(*tensors)
        return self.norm(tensors[0] + self.dropout(out))

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, dim_model)
        self.encoding.requires_grad = False
        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)
        _2i = torch.arange(0, dim_model, step=2).float()
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / dim_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / dim_model)))

    def forward(self, x):
        batch_size, max_len, d_model = x.size()
        out = self.encoding[:max_len, :].to("cuda:0")
        return out

def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask


class AttentionBiasHead(nn.Module):
    def __init__(self, dim_mlp_s:int, dim_in: int, dim_q: int, dim_k: int, dim_s: int, seq_len: int):  # dim_s is for static features
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)
        self.b = nn.Sequential(
            nn.Linear(dim_s, dim_mlp_s),
            nn.ReLU(),
            nn.Linear(dim_mlp_s, seq_len * seq_len),
        )
        self.q_cpe = nn.Linear(dim_q, dim_q)
        self.k_cpe = nn.Linear(dim_q, dim_q)
        self.dim_q = dim_q

    def forward(self, query, key, value, sf, atten_mask):
        b, l, d = query.shape # b表示batch_size
        query = self.q(query.reshape(b * l, d)).reshape(b, l, -1)
        key = self.k(key.reshape(b * l, d)).reshape(b, l, -1)
        value = self.v(value.reshape(b * l, d)).reshape(b, l, -1)


        bias = self.b(sf).reshape(b, l, l)

        temp = query.bmm(key.transpose(1, 2))
        scale = query.size(-1) ** 0.5
        atten_mask = atten_mask.to(torch.bool)
        scores = temp / scale
        scores.masked_fill_(atten_mask, 1e-9)
        softmax = f.softmax(scores, dim=-1) + bias
        out = softmax.bmm(value)

        att_seq = bias
        norm3 = torch.norm(att_seq[:, 1:, :] - att_seq[:, :-1, :], p=1) + torch.norm(att_seq[:,:,1:] - att_seq[:,:,:-1], p = 1)
        return out


class MultiHeadAttentionBias(nn.Module):
    def __init__(self, dim_mlp_s:int, num_heads: int, dim_in: int, dim_model: int, dim_q: int, dim_k: int, dim_s: int, seq_len: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionBiasHead(dim_mlp_s, dim_in, dim_q, dim_k, dim_s, seq_len) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_model)

    def forward(self, query, key, value, sf, atten_mask):
        results = [h(query, key, value, sf, atten_mask) for h in self.heads]
        hidden = torch.cat([result[0] for result in results], dim = -1)
        return self.linear(hidden)

class TransformerBiasEncoderLayer(nn.Module):
    def __init__(
            self,
            dim_mlp_s: int,
            dim_in: int = 256,
            dim_model: int = 256,
            num_heads: int = 2,
            dim_feedforward: int = 512,
            dropout: float = 0.1,
            dim_s: int = 19,
            seq_len: int = 30
    ):
        super().__init__()
        dim_q = dim_k = max(dim_model // num_heads, 1)
        self.attention = Residual(
            MultiHeadAttentionBias(dim_mlp_s, num_heads, dim_in, dim_model, dim_q, dim_k, dim_s, seq_len),
            dimension=dim_model,
            dropout=dropout
        )
        self.feed_forward = Residual(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout
        )

    def forward(self, src, sf, atten_mask):
        src = self.attention(src, src, src, sf, atten_mask)
        return self.feed_forward(src)


class TransformerBiasEncoder(nn.Module):
    def __init__(
            self,
            dim_mlp_s: int = 16,
            num_layers: int = 2,
            dim_in: int = 19,
            dim_model: int = 256,
            num_heads: int = 4,
            dim_feedforward: int = 512,
            dropout: float = 0.1,
            dim_s: int = 19,
            seq_len: int = 30
    ):
        super().__init__()
        self.first_layer = nn.Linear(dim_in, dim_model)
        self.PE = PositionalEncoding(dim_model, seq_len)
        dim_q, dim_k = dim_model//num_heads, dim_model//num_heads
        self.drop_out = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList(
            [
                TransformerBiasEncoderLayer(dim_mlp_s, dim_model, dim_model, num_heads, dim_feedforward, dropout, dim_s, seq_len)
                for _ in range(num_layers)
            ]
        )


    def forward(self, src, sf):
        lin_emb = self.first_layer(src)
        pos_emb = self.PE(src)
        src = lin_emb + pos_emb
        # src = self.drop_out(src)
        enc_look_ahead_mask = get_attn_subsequent_mask(src).to("cuda:0")
        for layer in self.layers:
            src = layer(src, sf, enc_look_ahead_mask)
        return src


class TransformerBiasNet(nn.Module):
    def __init__(
            self,
            dim_mlp_s:int,
            num_layers: int = 2,
            dim_in: int = 3,
            dim_model: int = 512,
            num_heads: int = 2,
            dim_feedforward: int = 1024,
            dropout: float = 0.1,
            dim_s: int = 4,
            seq_len: int = 30,
            dim_out: int = 3
    ):
        super().__init__()
        self.encoder = TransformerBiasEncoder(dim_mlp_s, num_layers, dim_in, dim_model, num_heads, dim_feedforward, dropout, dim_s,
                                              seq_len)
        self.fc = nn.Linear(dim_model, dim_out)

    def forward(self, src, sf):
        src = self.encoder(src, sf)
        out = self.fc(src[:, 0, :])
        return out

