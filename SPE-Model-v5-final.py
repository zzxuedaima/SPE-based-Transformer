from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# =========================
# Basic modules
# =========================

def feed_forward(dim_input: int = 16, dim_feedforward: int = 32) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedforward),
        nn.GELU(),
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
    def __init__(self, dim_model: int, max_len: int):
        super().__init__()
        encoding = torch.zeros(max_len, dim_model)
        pos = torch.arange(0, max_len).float().unsqueeze(dim=1)
        _2i = torch.arange(0, dim_model, step=2).float()
        encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / dim_model)))
        encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / dim_model)))
        self.register_buffer("encoding", encoding)

    def forward(self, x: Tensor) -> Tensor:
        _, seq_len, _ = x.size()
        return self.encoding[:seq_len, :].to(device=x.device, dtype=x.dtype)


def get_attn_subsequent_mask(seq: Tensor) -> Tensor:
    """Return True where future positions should be masked."""
    batch_size, seq_len, _ = seq.size()
    return torch.triu(
        torch.ones(batch_size, seq_len, seq_len, device=seq.device, dtype=torch.bool),
        diagonal=1,
    )


# =========================
# Stable SPE attention bias
# =========================

class LowRankStateBias(nn.Module):
    """
    Generate a bounded causal state-dependent L x L post-softmax attention bias
    from static features.

    The module uses a low-rank factorization:
        B_raw = R C^T / sqrt(rank) + diag(d),
    followed by a scaled tanh output and a causal lower-triangular constraint:
        B_SPE = beta * tanh(B_raw / beta) ⊙ L_causal.

    Thus, the SPE module itself handles both amplitude control and future-position
    suppression. No additional scaling or future-position masking is required in
    the attention forward pass.
    """

    def __init__(
        self,
        dim_s: int,
        dim_hidden: int,
        seq_len: int,
        rank: int = 4,
        max_bias: float = 2.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.rank = rank
        self.max_bias = max_bias

        self.backbone = nn.Sequential(
            nn.Linear(dim_s, dim_hidden),
            nn.GELU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.GELU(),
        )
        self.row = nn.Linear(dim_hidden, seq_len * rank)
        self.col = nn.Linear(dim_hidden, seq_len * rank)
        self.diag = nn.Linear(dim_hidden, seq_len)

        # Causal lower-triangular structure: only current and historical
        # positions are allowed to receive SPE modulation.
        lower_tri = torch.tril(torch.ones(seq_len, seq_len))
        self.register_buffer("lower_tri", lower_tri)

        # Start from a weak SPE perturbation for stable training.
        nn.init.zeros_(self.row.weight)
        nn.init.zeros_(self.row.bias)
        nn.init.zeros_(self.col.weight)
        nn.init.zeros_(self.col.bias)
        nn.init.zeros_(self.diag.weight)
        nn.init.zeros_(self.diag.bias)

    def forward(self, sf_static: Tensor, seq_len: Optional[int] = None) -> Tensor:
        if seq_len is None:
            seq_len = self.seq_len
        if seq_len != self.seq_len:
            raise ValueError(
                f"LowRankStateBias was initialized with seq_len={self.seq_len}, "
                f"but received seq_len={seq_len}."
            )

        b = sf_static.size(0)
        h = self.backbone(sf_static)
        row = self.row(h).view(b, self.seq_len, self.rank)
        col = self.col(h).view(b, self.seq_len, self.rank)
        raw_bias = row.bmm(col.transpose(1, 2)) / math.sqrt(max(self.rank, 1))

        diag = self.diag(h)
        eye = torch.eye(self.seq_len, device=sf_static.device, dtype=sf_static.dtype).unsqueeze(0)
        raw_bias = raw_bias + eye * diag.unsqueeze(1)

        # Bounded causal SPE output. The scaled tanh bounds each entry within
        # (-max_bias, max_bias), and the lower-triangular constraint prevents
        # future information leakage.
        bias = self.max_bias * torch.tanh(raw_bias / max(self.max_bias, 1e-6))
        causal = self.lower_tri.to(device=sf_static.device, dtype=sf_static.dtype).unsqueeze(0)
        bias = bias * causal
        return bias


class AttentionBiasHead(nn.Module):
    def __init__(
        self,
        dim_mlp_s: int,
        dim_in: int,
        dim_q: int,
        dim_k: int,
        dim_s: int,
        seq_len: int,
        bias_scale: float = 0.1,
        bias_rank: int = 4,
        max_bias: float = 2.0,
        scale_bias_by_len: bool = True,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

        # Low-rank bounded causal SPE bias.
        # For backward compatibility with the training script, the final SPE
        # amplitude bound is computed from the original bias_scale/max_bias
        # settings here, but this bound is embedded inside LowRankStateBias.
        # Thus no extra scaling is applied in the forward attention calculation.
        final_spe_bound = bias_scale * max_bias
        if scale_bias_by_len:
            final_spe_bound = final_spe_bound / math.sqrt(seq_len)
        self.final_spe_bound = final_spe_bound

        self.state_bias = LowRankStateBias(
            dim_s=dim_s,
            dim_hidden=dim_mlp_s,
            seq_len=seq_len,
            rank=bias_rank,
            max_bias=final_spe_bound,
        )

        self.seq_len = seq_len
        self.attn_dropout = nn.Dropout(attention_dropout)

        self.save_attention = False
        self.last_base_attn: Optional[Tensor] = None
        self.last_spe_attn: Optional[Tensor] = None
        self.last_spe_bias: Optional[Tensor] = None

    def set_save_attention(self, flag: bool = True) -> None:
        self.save_attention = flag

    def forward(self, query: Tensor, key: Tensor, value: Tensor, sf: Tensor, atten_mask: Tensor) -> Tensor:
        b, l, d = query.shape
        if l != self.seq_len:
            raise ValueError(
                f"Input sequence length l={l} does not match model seq_len={self.seq_len}. "
                "Please set max_len and seq_len consistently."
            )

        query = self.q(query.reshape(b * l, d)).reshape(b, l, -1)
        key = self.k(key.reshape(b * l, d)).reshape(b, l, -1)
        value = self.v(value.reshape(b * l, d)).reshape(b, l, -1)

        # Static state features: e0, Dr. They should be constant within one window.
        if sf.dim() == 3:
            sf_static = sf[:, 0, :]
        else:
            sf_static = sf

        raw_scores = query.bmm(key.transpose(1, 2)) / math.sqrt(query.size(-1))

        # Standard sequence-dependent attention weights.
        # base_attn is a row-normalized probability distribution produced by softmax.
        base_scores = raw_scores.masked_fill(atten_mask, -1e9)
        base_attn = F.softmax(base_scores, dim=-1)

        # Post-softmax SPE bias generated from static state parameters.
        # The returned bias is already bounded by the SPE module and constrained
        # to be lower triangular, so it cannot reintroduce future information.
        spe_bias = self.state_bias(sf_static, seq_len=l)

        # IMPORTANT: SPE is added AFTER softmax and BEFORE multiplication with V:
        #     O = (softmax(QK^T / sqrt(d_k) + M) + B_SPE) V
        # Hence spe_attn is a state-modulated aggregation matrix, not necessarily
        # a row-normalized probability distribution.
        spe_attn = base_attn + spe_bias
        spe_attn = self.attn_dropout(spe_attn)

        if self.save_attention:
            self.last_base_attn = base_attn.detach().cpu()
            self.last_spe_attn = spe_attn.detach().cpu()
            self.last_spe_bias = spe_bias.detach().cpu()

        out = spe_attn.bmm(value)
        return out


class MultiHeadAttentionBias(nn.Module):
    def __init__(
        self,
        dim_mlp_s: int,
        num_heads: int,
        dim_in: int,
        dim_model: int,
        dim_q: int,
        dim_k: int,
        dim_s: int,
        seq_len: int,
        bias_scale: float = 0.1,
        bias_rank: int = 4,
        max_bias: float = 2.0,
        scale_bias_by_len: bool = True,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                AttentionBiasHead(
                    dim_mlp_s=dim_mlp_s,
                    dim_in=dim_in,
                    dim_q=dim_q,
                    dim_k=dim_k,
                    dim_s=dim_s,
                    seq_len=seq_len,
                    bias_scale=bias_scale,
                    bias_rank=bias_rank,
                    max_bias=max_bias,
                    scale_bias_by_len=scale_bias_by_len,
                    attention_dropout=attention_dropout,
                )
                for _ in range(num_heads)
            ]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_model)

    def set_save_attention(self, flag: bool = True) -> None:
        for h in self.heads:
            h.set_save_attention(flag)

    def get_attention_maps(self) -> Dict[str, Tensor]:
        base, spe, bias = [], [], []
        for h in self.heads:
            if h.last_spe_attn is None:
                continue
            base.append(h.last_base_attn)
            spe.append(h.last_spe_attn)
            bias.append(h.last_spe_bias)
        if not spe:
            return {}
        return {
            "base_attn": torch.stack(base, dim=0),  # [heads, batch, L, L]
            "spe_attn": torch.stack(spe, dim=0),
            "spe_bias": torch.stack(bias, dim=0),
        }

    def forward(self, query: Tensor, key: Tensor, value: Tensor, sf: Tensor, atten_mask: Tensor) -> Tensor:
        results = [h(query, key, value, sf, atten_mask) for h in self.heads]
        hidden = torch.cat(results, dim=-1)
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
        seq_len: int = 30,
        bias_scale: float = 0.1,
        bias_rank: int = 4,
        max_bias: float = 2.0,
        scale_bias_by_len: bool = True,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        dim_q = dim_k = max(dim_model // num_heads, 1)
        self.attention = Residual(
            MultiHeadAttentionBias(
                dim_mlp_s=dim_mlp_s,
                num_heads=num_heads,
                dim_in=dim_in,
                dim_model=dim_model,
                dim_q=dim_q,
                dim_k=dim_k,
                dim_s=dim_s,
                seq_len=seq_len,
                bias_scale=bias_scale,
                bias_rank=bias_rank,
                max_bias=max_bias,
                scale_bias_by_len=scale_bias_by_len,
                attention_dropout=attention_dropout,
            ),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = Residual(feed_forward(dim_model, dim_feedforward), dimension=dim_model, dropout=dropout)

    def set_save_attention(self, flag: bool = True) -> None:
        self.attention.sublayer.set_save_attention(flag)

    def get_attention_maps(self) -> Dict[str, Tensor]:
        return self.attention.sublayer.get_attention_maps()

    def forward(self, src: Tensor, sf: Tensor, atten_mask: Tensor) -> Tensor:
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
        seq_len: int = 30,
        bias_scale: float = 0.1,
        bias_rank: int = 4,
        max_bias: float = 2.0,
        scale_bias_by_len: bool = True,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.first_layer = nn.Linear(dim_in, dim_model)
        self.PE = PositionalEncoding(dim_model, seq_len)
        self.drop_out = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList(
            [
                TransformerBiasEncoderLayer(
                    dim_mlp_s=dim_mlp_s,
                    dim_in=dim_model,
                    dim_model=dim_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    dim_s=dim_s,
                    seq_len=seq_len,
                    bias_scale=bias_scale,
                    bias_rank=bias_rank,
                    max_bias=max_bias,
                    scale_bias_by_len=scale_bias_by_len,
                    attention_dropout=attention_dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def set_save_attention(self, flag: bool = True) -> None:
        for layer in self.layers:
            layer.set_save_attention(flag)

    def get_last_attention_maps(self) -> Dict[str, Tensor]:
        if len(self.layers) == 0:
            return {}
        return self.layers[-1].get_attention_maps()

    def forward(self, src: Tensor, sf: Tensor) -> Tensor:
        src = self.first_layer(src) + self.PE(src)
        src = self.drop_out(src)
        mask = get_attn_subsequent_mask(src)
        for layer in self.layers:
            src = layer(src, sf, mask)
        return src


class TransformerBiasNet(nn.Module):
    def __init__(
        self,
        dim_mlp_s: int,
        num_layers: int = 2,
        dim_in: int = 3,
        dim_model: int = 512,
        num_heads: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        dim_s: int = 4,
        seq_len: int = 30,
        dim_out: int = 3,
        bias_scale: float = 0.1,
        bias_rank: int = 4,
        max_bias: float = 2.0,
        scale_bias_by_len: bool = True,
        attention_dropout: float = 0.0,
        output_activation: Optional[str] = "sigmoid",
    ):
        super().__init__()
        self.encoder = TransformerBiasEncoder(
            dim_mlp_s=dim_mlp_s,
            num_layers=num_layers,
            dim_in=dim_in,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dim_s=dim_s,
            seq_len=seq_len,
            bias_scale=bias_scale,
            bias_rank=bias_rank,
            max_bias=max_bias,
            scale_bias_by_len=scale_bias_by_len,
            attention_dropout=attention_dropout,
        )
        self.fc = nn.Linear(dim_model, dim_out)
        self.output_activation = output_activation

    def set_save_attention(self, flag: bool = True) -> None:
        self.encoder.set_save_attention(flag)

    def get_last_attention_maps(self) -> Dict[str, Tensor]:
        return self.encoder.get_last_attention_maps()

    def forward(self, src: Tensor, sf: Tensor) -> Tensor:
        src = self.encoder(src, sf)
        out = self.fc(src)

        # [v2 修改4] MinMax-normalized targets are in [0, 1]. Sigmoid suppresses
        # local spikes caused by extrapolated normalized outputs. Set
        # output_activation=None if extrapolation outside the training range is needed.
        if self.output_activation is not None:
            act = self.output_activation.lower()
            if act == "sigmoid":
                out = torch.sigmoid(out)
            elif act == "clamp":
                out = torch.clamp(out, 0.0, 1.0)
            else:
                raise ValueError(f"Unknown output_activation: {self.output_activation}")
        return out
