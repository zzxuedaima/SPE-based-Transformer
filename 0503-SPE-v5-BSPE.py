from __future__ import annotations

import importlib.util
import math
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt


# =========================
# Import model from hyphenated file name
# =========================
MODEL_PATH = Path(__file__).with_name("SPE-Model-v5-final.py")
spec = importlib.util.spec_from_file_location("SPE_Model_v5_final", MODEL_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load model from {MODEL_PATH}")
_model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_model_module)
TransformerBiasNet = _model_module.TransformerBiasNet


# =========================
# Reproducibility
# =========================
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


# =========================
# User settings
# =========================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data structure:
# columns: e0, Dr, q_{i-1}, p_{i-1}, e_{i-1}, q_i, p_i, e_i
column_total = 8
in_loc = 5
s_loc = 2
p_loc = 5
q_loc = 6
s1_loc = 7

# Model parameters
max_len = 10          # You can test 10, 50, 100, 200, etc.
window_stride = max(1, max_len // 2)  # Overlapping windows. Use max_len for non-overlap.

batch_size = 512       # Smaller than 4096 is safer for large max_len.
epoch = 1000
warmup_epochs = 200
learning_rate = 5e-4

# Transformer
num_layers = 1
num_heads = 1
dim_in = 3
dim_s = 2
dim_out = 3
dim_model = 128
dim_feedforward = 128
dim_mlp_s = 16
dropout = 0.1
attention_dropout = 0.0

# SPE-bias stabilization
bias_scale = 0.1       # Effective scale is bias_scale / sqrt(max_len) inside the model.
bias_rank = 4          # Low-rank SPE bias. Increase only if attention patterns look too simple.
max_bias = 2.0
scale_bias_by_len = True
output_activation = "sigmoid"  # Keeps normalized outputs in [0,1]. Use None if extrapolation is required.

# Loss stabilization
smooth_weight = 0.05       # First-difference loss weight.
curvature_weight = 0.01    # Second-difference loss weight.
grad_clip_norm = 1.0

PLOT_WITH_TRUE_X = True
CLIP_PREDICTION_TO_SCALER_RANGE = True


# =========================
# Plot helpers
# =========================
def figure(x_act, x_pre, y_act, y_pre, title=None):
    fig = plt.figure(figsize=(6, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(x_act, y_act, label="Actual")
    ax.plot(x_pre, y_pre, label="Prediction")
    ax.legend()
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    if title is not None:
        ax.set_title(title)
    ax.grid()
    fig.tight_layout(pad=0)
    return fig


def pick_prediction_x(x_actual, x_prediction):
    return x_actual if PLOT_WITH_TRUE_X else x_prediction


# =========================
# Window utilities
# =========================
def get_window_weights(length: int) -> np.ndarray:
    """Smooth overlap-add weights; endpoints are nonzero."""
    if length <= 1:
        return np.ones(length, dtype=np.float32)
    weights = np.hanning(length + 2)[1:-1].astype(np.float32)
    weights = weights / weights.mean()
    return weights


def detect_segments_by_static_features(
    data_real: np.ndarray,
    static_cols: Sequence[int] = (0, 1),
    atol: float = 1e-10,
) -> List[Tuple[int, int]]:
    """
    Detect continuous segments by changes in static features e0 and Dr.

    This prevents windows from crossing obvious state boundaries. If your file contains
    an explicit specimen/cycle ID, replacing this function with ID-based splitting is better.
    """
    if len(data_real) == 0:
        return []
    static = data_real[:, list(static_cols)]
    change = np.any(np.abs(static[1:] - static[:-1]) > atol, axis=1)
    cuts = np.where(change)[0] + 1
    bounds = [0] + cuts.tolist() + [len(data_real)]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1) if bounds[i + 1] > bounds[i]]


def add_known_cuts_to_segments(
    segments: List[Tuple[int, int]], cuts: Iterable[int], n_total: int
) -> List[Tuple[int, int]]:
    """Add known test boundaries such as row1 and row2 without crossing them."""
    valid_cuts = sorted({int(c) for c in cuts if 0 < int(c) < n_total})
    if not valid_cuts:
        return segments
    new_segments = []
    for start, end in segments:
        internal = [c for c in valid_cuts if start < c < end]
        bounds = [start] + internal + [end]
        new_segments.extend((bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1))
    return [(s, e) for s, e in new_segments if e > s]


def make_windows_from_segments(
    data_scaled: np.ndarray,
    segments: Sequence[Tuple[int, int]],
    length: int,
    stride: int,
    include_tail: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build windows without crossing segment boundaries.

    Returns
    -------
    windows: [num_windows, length, num_features]
    indices: [num_windows, length], original row indices for each window
    """
    windows, indices = [], []
    for start, end in segments:
        seg_len = end - start
        if seg_len < length:
            continue

        starts = list(range(start, end - length + 1, stride))
        if include_tail:
            tail_start = end - length
            if len(starts) == 0 or starts[-1] != tail_start:
                starts.append(tail_start)

        for s in starts:
            idx = np.arange(s, s + length)
            windows.append(data_scaled[idx, :])
            indices.append(idx)

    if len(windows) == 0:
        raise ValueError(
            "No windows were created. Check max_len, window_stride, and segment lengths."
        )
    return np.stack(windows, axis=0), np.stack(indices, axis=0)


def reconstruct_by_overlap_average(
    window_predictions: np.ndarray,
    window_indices: np.ndarray,
    n_rows: int,
    weights: np.ndarray,
) -> np.ndarray:
    """Reconstruct row-wise predictions by weighted overlap averaging."""
    dim_out = window_predictions.shape[-1]
    pred_sum = np.zeros((n_rows, dim_out), dtype=np.float64)
    weight_sum = np.zeros((n_rows, 1), dtype=np.float64)
    w = weights.reshape(1, -1, 1).astype(np.float64)

    for pred, idx in zip(window_predictions, window_indices):
        pred_sum[idx, :] += pred * w[0]
        weight_sum[idx, :] += w[0]

    out = np.full((n_rows, dim_out), np.nan, dtype=np.float64)
    covered = weight_sum[:, 0] > 0
    out[covered, :] = pred_sum[covered, :] / weight_sum[covered, :]

    if not np.all(covered):
        # Fill any uncovered rows using nearest available prediction.
        covered_idx = np.where(covered)[0]
        if len(covered_idx) == 0:
            raise ValueError("No rows were covered by prediction windows.")
        for i in np.where(~covered)[0]:
            nearest = covered_idx[np.argmin(np.abs(covered_idx - i))]
            out[i, :] = out[nearest, :]
    return out


# =========================
# Loss and prediction helpers
# =========================
def initialize_weights(m):
    if hasattr(m, "weight") and hasattr(m.weight, "dim") and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    # weights: [1, L, 1]
    se = (pred - target) ** 2
    return (se * weights).sum() / (weights.sum() * pred.shape[0] * pred.shape[-1])


def sequence_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    point_weight: float = 1.0,
    smooth_weight_: float = 0.05,
    curvature_weight_: float = 0.01,
) -> torch.Tensor:
    loss = point_weight * weighted_mse(pred, target, weights)

    if pred.size(1) >= 2 and smooth_weight_ > 0:
        dp = pred[:, 1:, :] - pred[:, :-1, :]
        dt = target[:, 1:, :] - target[:, :-1, :]
        w1 = weights[:, 1:, :]
        loss = loss + smooth_weight_ * weighted_mse(dp, dt, w1)

    if pred.size(1) >= 3 and curvature_weight_ > 0:
        cp = pred[:, 2:, :] - 2 * pred[:, 1:-1, :] + pred[:, :-2, :]
        ct = target[:, 2:, :] - 2 * target[:, 1:-1, :] + target[:, :-2, :]
        w2 = weights[:, 1:-1, :]
        loss = loss + curvature_weight_ * weighted_mse(cp, ct, w2)

    return loss


class CustomLRScheduler(optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, warmup_epochs, T_max, lr_init, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max
        self.lr_init = lr_init
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [self.lr_init * (self.last_epoch + 1) / self.warmup_epochs for _ in self.base_lrs]
        ep = self.last_epoch - self.warmup_epochs
        denom = max(self.T_max - self.warmup_epochs, 1)
        return [0.5 * base_lr * (1 + math.cos(math.pi * ep / denom)) for base_lr in self.base_lrs]


def predict_windows(
    model: nn.Module,
    windows: np.ndarray,
    batch_size_: int,
    device_: torch.device,
) -> np.ndarray:
    dataset = TensorDataset(torch.from_numpy(windows.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=batch_size_, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for (item,) in loader:
            item = item.to(device_)
            pred = model(src=item[:, :, s_loc:in_loc], sf=item[:, :, :s_loc])
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds, axis=0)


def predict_full_curve(
    model: nn.Module,
    data_scaled: np.ndarray,
    segments: Sequence[Tuple[int, int]],
    length: int,
    stride: int,
    weights: np.ndarray,
    batch_size_: int,
    device_: torch.device,
) -> np.ndarray:
    windows, indices = make_windows_from_segments(
        data_scaled, segments, length=length, stride=stride, include_tail=True
    )
    pred_windows = predict_windows(model, windows, batch_size_, device_)
    pred = reconstruct_by_overlap_average(pred_windows, indices, len(data_scaled), weights)
    if CLIP_PREDICTION_TO_SCALER_RANGE:
        pred = np.clip(pred, 0.0, 1.0)
    return pred


def calculate_metrics(target, predict):
    target = np.asarray(target)
    predict = np.asarray(predict)
    mae = np.mean(np.abs(target - predict))
    sse = np.sum((target - predict) ** 2)
    sst = np.sum((target - np.mean(target)) ** 2)
    r2 = np.nan if sst == 0 else 1 - sse / sst
    non_zero = target != 0
    if np.any(non_zero):
        mape = np.mean(np.abs((target[non_zero] - predict[non_zero]) / target[non_zero])) * 100
    else:
        mape = np.nan
    return mae, mape, r2


def plot_time_series_diagnostics(Data_test_real, inversed_out_test, start, end, title):
    fig = plt.figure(figsize=(9, 4))
    ax = fig.add_subplot(111)
    ax.plot(Data_test_real[start:end, q_loc], label="q actual")
    ax.plot(inversed_out_test[start:end, q_loc], label="q prediction")
    ax.set_title(f"{title}: q-time diagnostic")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    return fig


# Optional: interval-averaged attention map for reviewer response.
def extract_interval_mean_attention(
    model: nn.Module,
    data_scaled: np.ndarray,
    interval: Tuple[int, int],
    length: int,
    stride: int,
    batch_size_: int,
    device_: torch.device,
    attn_key: str = "spe_attn",
) -> np.ndarray:
    """
    Extract mean attention map over a selected interval.

    Returns a [length, length] matrix averaged over windows, batch samples, and heads.
    """
    windows, _ = make_windows_from_segments(
        data_scaled, [interval], length=length, stride=stride, include_tail=True
    )
    dataset = TensorDataset(torch.from_numpy(windows.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=batch_size_, shuffle=False)
    maps = []
    model.eval()
    model.set_save_attention(True)
    with torch.no_grad():
        for (item,) in loader:
            item = item.to(device_)
            _ = model(src=item[:, :, s_loc:in_loc], sf=item[:, :, :s_loc])
            attn = model.get_last_attention_maps()
            if attn_key not in attn:
                raise RuntimeError(f"Attention key {attn_key} not found. Available: {list(attn.keys())}")
            # [heads, batch, L, L] -> [L, L]
            maps.append(attn[attn_key].mean(dim=(0, 1)).numpy())
    model.set_save_attention(False)
    return np.mean(np.stack(maps, axis=0), axis=0)


def plot_attention_map(A: np.ndarray, title: str):
    fig = plt.figure(figsize=(5.5, 4.8))
    ax = fig.add_subplot(111)
    im = ax.imshow(A, aspect="auto", origin="upper")
    ax.set_xlabel("Historical loading step / key position")
    ax.set_ylabel("Current loading step / query position")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Post-softmax SPE-modulated aggregation weight")
    fig.tight_layout()
    return fig




# =========================
# All-point attention extraction for testing data
# =========================
def extract_all_point_attention_table(
    model: nn.Module,
    data_scaled: np.ndarray,
    specimen_name: str,
    start: int,
    end: int,
    length: int,
    device_: torch.device,
    attn_key: str = "spe_attn",
    n_lags: int = 10,
    batch_size_: int = 512,
) -> pd.DataFrame:
    """
    Extract attention vectors for every point in one testing segment.

    Output table:
        rows    = every data point in [start, end)
        columns = Specimen, LocalIndex, TestIndex, ExcelRow, T0, T1, ..., T9

    Notes
    -----
    T0 is the selected/current loading point.
    T1 is one step before the selected point.
    T2 is two steps before the selected point, and so on.

    For the first several points in a segment, there are not enough previous
    loading states. Those unavailable lag positions are filled with NaN.
    """
    if end <= start:
        raise ValueError(f"Invalid interval for {specimen_name}: start={start}, end={end}")

    if end - start < length:
        raise ValueError(
            f"Segment {specimen_name} is shorter than max_len={length}: "
            f"start={start}, end={end}."
        )

    windows = []
    query_positions = []
    test_indices = []
    local_indices = []
    excel_rows = []

    for target_idx in range(start, end):
        # Build a length=max_len window containing the selected point.
        # For most points, the window ends at the selected point.
        # For the first length-1 points in a segment, the window starts from
        # the segment start and the selected point is placed at query_pos < L-1.
        if target_idx < start + length - 1:
            win_start = start
            query_pos = target_idx - win_start
        else:
            win_start = target_idx - length + 1
            query_pos = length - 1

        win_end = win_start + length

        # Safety check: do not cross the segment boundary.
        if win_start < start or win_end > end:
            continue

        item_np = data_scaled[win_start:win_end, :].astype(np.float32)

        windows.append(item_np)
        query_positions.append(query_pos)
        test_indices.append(target_idx)          # 0-based row index in Data_test_real/Data_test
        local_indices.append(target_idx - start) # 0-based row index within this specimen/segment

        # Data_test_real = Data_test_import.iloc[1:, :]
        # Therefore, Data_test_real index 0 corresponds to Excel row 2.
        excel_rows.append(target_idx + 2)

    if len(windows) == 0:
        raise ValueError(
            f"No valid attention windows were created for {specimen_name}. "
            f"Check start={start}, end={end}, and length={length}."
        )

    windows = np.stack(windows, axis=0)
    query_positions = np.asarray(query_positions, dtype=int)

    dataset = TensorDataset(torch.from_numpy(windows.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=batch_size_, shuffle=False)

    records = []
    model.eval()
    model.set_save_attention(True)

    offset = 0
    try:
        with torch.no_grad():
            for (item,) in loader:
                item = item.to(device_)

                _ = model(src=item[:, :, s_loc:in_loc], sf=item[:, :, :s_loc])

                attn_maps = model.get_last_attention_maps()
                if attn_key not in attn_maps:
                    raise RuntimeError(
                        f"Attention key '{attn_key}' not found. "
                        f"Available keys: {list(attn_maps.keys())}"
                    )

                # attn shape: [heads, batch, L, L].
                # Average over heads -> [batch, L, L].
                attn = attn_maps[attn_key].mean(dim=0).cpu().numpy()
                batch_n = item.shape[0]

                for b in range(batch_n):
                    global_i = offset + b
                    qpos = query_positions[global_i]

                    A = attn[b]              # [L, L]
                    attn_row = A[qpos, :]    # attention row for the selected/current point

                    # Keep only the accessible current and historical positions.
                    # Reverse the order so that:
                    # T0 = current point, T1 = one step before, ..., T9 = nine steps before.
                    accessible = attn_row[: qpos + 1][::-1]

                    values = np.full(n_lags, np.nan, dtype=float)
                    m = min(n_lags, len(accessible))
                    values[:m] = accessible[:m]

                    record = {
                        "Specimen": specimen_name,
                        "LocalIndex": local_indices[global_i],
                        "TestIndex": test_indices[global_i],
                        "ExcelRow": excel_rows[global_i],
                    }
                    for k in range(n_lags):
                        record[f"T{k}"] = values[k]
                    records.append(record)

                offset += batch_n
    finally:
        model.set_save_attention(False)

    return pd.DataFrame(records)


def export_all_test_attention_tables(
    model: nn.Module,
    data_scaled: np.ndarray,
    row1: int,
    row2: int,
    length: int,
    device_: torch.device,
    output_file: str = "0503-SPE-v5-PostSoftmax-AllPoint-Attention-Testing-T0T9.xlsx",
    attn_key: str = "spe_attn",
    n_lags: int = 10,
    batch_size_: int = 512,
) -> None:
    """
    Export all-point attention tables for the full testing dataset.

    The workbook contains one sheet for the full testing dataset and one sheet
    for each testing segment:
        UT1     : Data_test[:row1]
        TCUI-4  : Data_test[row1:row2]
        TCUI-21 : Data_test[row2:]
    """
    segment_specs = [
        ("UT1", 0, row1),
        ("TCUI-4", row1, row2),
        ("TCUI-21", row2, len(data_scaled)),
    ]

    segment_tables = {}
    for name, start, end in segment_specs:
        print(f"Extracting all-point attention for {name}: rows [{start}, {end})")
        segment_tables[name] = extract_all_point_attention_table(
            model=model,
            data_scaled=data_scaled,
            specimen_name=name,
            start=start,
            end=end,
            length=length,
            device_=device_,
            attn_key=attn_key,
            n_lags=n_lags,
            batch_size_=batch_size_,
        )

    all_table = pd.concat(segment_tables.values(), axis=0, ignore_index=True)

    with pd.ExcelWriter(output_file) as writer:
        all_table.to_excel(writer, sheet_name="All_Test", index=False)
        for name, table in segment_tables.items():
            table.to_excel(writer, sheet_name=name, index=False)

    print(f"Saved all-point attention tables to {output_file}")




# =========================
# All-point B_SPE correction extraction for testing data
# =========================
def _attention_tensor_to_batch_matrix(attn_obj, batch_n: int, name: str) -> np.ndarray:
    """
    Convert a saved attention/correction tensor to [batch, L, L].

    Supported input shapes:
        [heads, batch, L, L] -> averaged over heads
        [batch, L, L]        -> unchanged
        [L, L]               -> repeated for the batch

    This helper is used because different model implementations may save
    B_SPE with slightly different dimensions.
    """
    if torch.is_tensor(attn_obj):
        arr = attn_obj.detach().cpu().numpy()
    else:
        arr = np.asarray(attn_obj)

    if arr.ndim == 4:
        # [heads, batch, L, L] -> [batch, L, L]
        arr = arr.mean(axis=0)
    elif arr.ndim == 3:
        # Usually [batch, L, L]. If it is [heads, L, L], average and repeat.
        if arr.shape[0] != batch_n:
            arr = arr.mean(axis=0)
            arr = np.repeat(arr[None, :, :], batch_n, axis=0)
    elif arr.ndim == 2:
        # [L, L] -> [batch, L, L]
        arr = np.repeat(arr[None, :, :], batch_n, axis=0)
    else:
        raise ValueError(
            f"Unsupported shape for '{name}': {arr.shape}. Expected [H,B,L,L], [B,L,L], or [L,L]."
        )

    if arr.shape[0] != batch_n:
        raise ValueError(
            f"Batch size mismatch for '{name}': got shape {arr.shape}, expected batch_n={batch_n}."
        )
    return arr


def get_bspe_batch_from_attention_maps(
    attn_maps: dict,
    batch_n: int,
    bspe_key: str = "spe_bias",
    spe_attn_key: str = "spe_attn",
    base_attn_keys: Sequence[str] = (
        "base_attn", "hist_attn", "softmax_attn", "vanilla_attn", "raw_attn", "attn"
    ),
    bspe_key_candidates: Sequence[str] = (
        "spe_bias", "B_SPE", "bspe", "spe_correction", "state_bias", "post_softmax_bias", "bias"
    ),
) -> np.ndarray:
    """
    Return B_SPE as [batch, L, L].

    Priority:
    1. Use the directly saved B_SPE tensor if available.
    2. If both final SPE attention and original/base attention are saved,
       compute B_SPE = SPE attention - base attention.

    If neither is available, the model file must be modified to save B_SPE
    inside get_last_attention_maps().
    """
    available = list(attn_maps.keys())

    # 1. Direct extraction of B_SPE.
    direct_keys = [bspe_key] + [k for k in bspe_key_candidates if k != bspe_key]
    for key in direct_keys:
        if key in attn_maps:
            return _attention_tensor_to_batch_matrix(attn_maps[key], batch_n=batch_n, name=key)

    # 2. Fallback: compute B_SPE as final SPE attention minus original/base attention.
    final_key = spe_attn_key if spe_attn_key in attn_maps else None
    if final_key is None:
        for key in ("spe_attn", "final_attn", "attn_spe", "spe_modulated_attn"):
            if key in attn_maps:
                final_key = key
                break

    base_key = None
    for key in base_attn_keys:
        if key in attn_maps:
            base_key = key
            break

    if final_key is not None and base_key is not None:
        final_attn = _attention_tensor_to_batch_matrix(attn_maps[final_key], batch_n=batch_n, name=final_key)
        base_attn = _attention_tensor_to_batch_matrix(attn_maps[base_key], batch_n=batch_n, name=base_key)
        return final_attn - base_attn

    raise RuntimeError(
        "B_SPE was not found in model.get_last_attention_maps(). "
        f"Available keys are: {available}. "
        "Please save the SPE correction term in the model, for example as "
        "last_attention_maps['spe_bias'] = B_SPE.detach().cpu(). "
        "Alternatively, save both the original softmax attention and the final SPE attention so that "
        "B_SPE can be computed as their difference."
    )


def extract_all_point_bspe_table(
    model: nn.Module,
    data_scaled: np.ndarray,
    specimen_name: str,
    start: int,
    end: int,
    length: int,
    device_: torch.device,
    bspe_key: str = "spe_bias",
    spe_attn_key: str = "spe_attn",
    n_lags: int = 10,
    batch_size_: int = 512,
) -> pd.DataFrame:
    """
    Extract B_SPE correction vectors for every point in one testing segment.

    Output table:
        rows    = every data point in [start, end)
        columns = Specimen, LocalIndex, TestIndex, ExcelRow, T0, T1, ..., T9

    T0 is the nearest available input position for the selected prediction point.
    T1 is one step before it, and so on.

    The extracted values are from B_SPE, not from the final SPE-attention map.
    """
    if end <= start:
        raise ValueError(f"Invalid interval for {specimen_name}: start={start}, end={end}")

    if end - start < length:
        raise ValueError(
            f"Segment {specimen_name} is shorter than max_len={length}: "
            f"start={start}, end={end}."
        )

    windows = []
    query_positions = []
    test_indices = []
    local_indices = []
    excel_rows = []

    for target_idx in range(start, end):
        if target_idx < start + length - 1:
            win_start = start
            query_pos = target_idx - win_start
        else:
            win_start = target_idx - length + 1
            query_pos = length - 1

        win_end = win_start + length
        if win_start < start or win_end > end:
            continue

        item_np = data_scaled[win_start:win_end, :].astype(np.float32)
        windows.append(item_np)
        query_positions.append(query_pos)
        test_indices.append(target_idx)
        local_indices.append(target_idx - start)
        excel_rows.append(target_idx + 2)

    if len(windows) == 0:
        raise ValueError(
            f"No valid B_SPE windows were created for {specimen_name}. "
            f"Check start={start}, end={end}, and length={length}."
        )

    windows = np.stack(windows, axis=0)
    query_positions = np.asarray(query_positions, dtype=int)

    dataset = TensorDataset(torch.from_numpy(windows.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=batch_size_, shuffle=False)

    records = []
    model.eval()
    model.set_save_attention(True)

    offset = 0
    try:
        with torch.no_grad():
            for (item,) in loader:
                item = item.to(device_)
                _ = model(src=item[:, :, s_loc:in_loc], sf=item[:, :, :s_loc])

                attn_maps = model.get_last_attention_maps()
                bspe_batch = get_bspe_batch_from_attention_maps(
                    attn_maps=attn_maps,
                    batch_n=item.shape[0],
                    bspe_key=bspe_key,
                    spe_attn_key=spe_attn_key,
                )

                batch_n = item.shape[0]
                for b in range(batch_n):
                    global_i = offset + b
                    qpos = query_positions[global_i]

                    B = bspe_batch[b]       # [L, L]
                    bspe_row = B[qpos, :]   # B_SPE row for the selected prediction point

                    # Keep only accessible current/previous positions and reverse order:
                    # T0 = nearest input position, T1 = one step before, ..., T9 = nine steps before.
                    accessible = bspe_row[: qpos + 1][::-1]

                    values = np.full(n_lags, np.nan, dtype=float)
                    m = min(n_lags, len(accessible))
                    values[:m] = accessible[:m]

                    record = {
                        "Specimen": specimen_name,
                        "LocalIndex": local_indices[global_i],
                        "TestIndex": test_indices[global_i],
                        "ExcelRow": excel_rows[global_i],
                    }
                    for k in range(n_lags):
                        record[f"T{k}"] = values[k]
                    records.append(record)

                offset += batch_n
    finally:
        model.set_save_attention(False)

    return pd.DataFrame(records)


def export_all_test_bspe_tables(
    model: nn.Module,
    data_scaled: np.ndarray,
    row1: int,
    row2: int,
    length: int,
    device_: torch.device,
    output_file: str = "0503-SPE-v5-AllPoint-BSPE-Testing-T0T9.xlsx",
    bspe_key: str = "spe_bias",
    spe_attn_key: str = "spe_attn",
    n_lags: int = 10,
    batch_size_: int = 512,
) -> None:
    """
    Export all-point B_SPE correction tables for the full testing dataset.

    The workbook contains one sheet for the full testing dataset and one sheet
    for each testing segment:
        UT1     : Data_test[:row1]
        TCUI-4  : Data_test[row1:row2]
        TCUI-21 : Data_test[row2:]
    """
    segment_specs = [
        ("UT1", 0, row1),
        ("TCUI-4", row1, row2),
        ("TCUI-21", row2, len(data_scaled)),
    ]

    segment_tables = {}
    for name, start, end in segment_specs:
        print(f"Extracting all-point B_SPE for {name}: rows [{start}, {end})")
        segment_tables[name] = extract_all_point_bspe_table(
            model=model,
            data_scaled=data_scaled,
            specimen_name=name,
            start=start,
            end=end,
            length=length,
            device_=device_,
            bspe_key=bspe_key,
            spe_attn_key=spe_attn_key,
            n_lags=n_lags,
            batch_size_=batch_size_,
        )

    all_table = pd.concat(segment_tables.values(), axis=0, ignore_index=True)

    with pd.ExcelWriter(output_file) as writer:
        all_table.to_excel(writer, sheet_name="All_Test", index=False)
        for name, table in segment_tables.items():
            table.to_excel(writer, sheet_name=name, index=False)

    print(f"Saved all-point B_SPE tables to {output_file}")

# =========================
# Main
# =========================
if __name__ == "__main__":
    Data_train_import = pd.read_excel("Training.xlsx", header=None)
    Data_test_import = pd.read_excel("Testing.xlsx", header=None)
    Data_train_real = Data_train_import.iloc[1:, :].values[:, :].astype("float64")
    Data_test_real = Data_test_import.iloc[1:, :].values[:, :].astype("float64")

    # Use one scaler fitted on training data only.
    Scaler = MinMaxScaler(feature_range=(0, 1))
    Data_train = Scaler.fit_transform(Data_train_real)
    Data_test = Scaler.transform(Data_test_real)

    # Known test boundaries used in your original plotting.
    row1 = 9661
    row2 = 15032

    train_segments = detect_segments_by_static_features(Data_train_real, static_cols=(0, 1))
    test_segments = detect_segments_by_static_features(Data_test_real, static_cols=(0, 1))
    test_segments = add_known_cuts_to_segments(test_segments, cuts=[row1, row2], n_total=len(Data_test_real))

    print(f"Detected train segments: {len(train_segments)}")
    print(f"Detected test segments:  {len(test_segments)}")

    window_weights_np = get_window_weights(max_len)
    window_weights_torch = torch.tensor(window_weights_np, dtype=torch.float32, device=device).view(1, max_len, 1)

    train_windows, train_indices = make_windows_from_segments(
        Data_train, train_segments, length=max_len, stride=window_stride, include_tail=True
    )
    test_windows, test_indices = make_windows_from_segments(
        Data_test, test_segments, length=max_len, stride=window_stride, include_tail=True
    )

    print(f"Train windows: {train_windows.shape}")
    print(f"Test windows:  {test_windows.shape}")

    train_dataset = TensorDataset(torch.from_numpy(train_windows.astype(np.float32)))
    test_dataset = TensorDataset(torch.from_numpy(test_windows.astype(np.float32)))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = TransformerBiasNet(
        dim_mlp_s=dim_mlp_s,
        num_layers=num_layers,
        dim_in=dim_in,
        dim_model=dim_model,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        dim_s=dim_s,
        seq_len=max_len,
        dim_out=dim_out,
        bias_scale=bias_scale,
        bias_rank=bias_rank,
        max_bias=max_bias,
        scale_bias_by_len=scale_bias_by_len,
        attention_dropout=attention_dropout,
        output_activation=output_activation,
    )
    model.apply(initialize_weights)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-8, weight_decay=1e-6)
    scheduler = CustomLRScheduler(optimizer, warmup_epochs=warmup_epochs, T_max=epoch, lr_init=learning_rate)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    Loss_train, Loss_test = [], []

    for epoch_idx in range(epoch):
        print(f"Epochs = {epoch_idx + 1}")
        model.train()
        train_losses = []

        for (item,) in train_loader:
            item = item.to(device)
            target = item[:, :, in_loc:]

            optimizer.zero_grad()
            pred = model(src=item[:, :, s_loc:in_loc], sf=item[:, :, :s_loc])
            loss = sequence_loss(
                pred,
                target,
                window_weights_torch,
                smooth_weight_=smooth_weight,
                curvature_weight_=curvature_weight,
            )
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()
        Loss_train.append(float(np.mean(train_losses)))

        model.eval()
        test_losses = []
        with torch.no_grad():
            for (item,) in test_loader:
                item = item.to(device)
                target = item[:, :, in_loc:]
                pred = model(src=item[:, :, s_loc:in_loc], sf=item[:, :, :s_loc])
                loss = sequence_loss(
                    pred,
                    target,
                    window_weights_torch,
                    smooth_weight_=smooth_weight,
                    curvature_weight_=curvature_weight,
                )
                test_losses.append(loss.item())
        Loss_test.append(float(np.mean(test_losses)))

    Loss = np.column_stack((Loss_train, Loss_test))

    # Full-curve prediction by overlap averaging.
    y_train_pred = predict_full_curve(
        model, Data_train, train_segments, max_len, window_stride, window_weights_np, batch_size, device
    )
    y_test_pred = predict_full_curve(
        model, Data_test, test_segments, max_len, window_stride, window_weights_np, batch_size, device
    )

    prediction_train = np.concatenate((Data_train[:, :in_loc], y_train_pred), axis=1)
    prediction_test = np.concatenate((Data_test[:, :in_loc], y_test_pred), axis=1)

    inversed_out_train = Scaler.inverse_transform(prediction_train)
    inversed_out_test = Scaler.inverse_transform(prediction_test)

    print("\nRange check after inverse transform:")
    print(f"Actual test p range: {Data_test_real[:, 5].min():.6g}, {Data_test_real[:, 5].max():.6g}")
    print(f"Pred   test p range: {inversed_out_test[:, 5].min():.6g}, {inversed_out_test[:, 5].max():.6g}")
    print(f"Actual test q range: {Data_test_real[:, 6].min():.6g}, {Data_test_real[:, 6].max():.6g}")
    print(f"Pred   test q range: {inversed_out_test[:, 6].min():.6g}, {inversed_out_test[:, 6].max():.6g}")
    print(f"Actual test s1 range: {Data_test_real[:, 7].min():.6g}, {Data_test_real[:, 7].max():.6g}")
    print(f"Pred   test s1 range: {inversed_out_test[:, 7].min():.6g}, {inversed_out_test[:, 7].max():.6g}")

    # Metrics
    y_act_train = Data_train_real[:, in_loc:].astype("float64")
    y_act_test = Data_test_real[:, in_loc:].astype("float64")

    train_p = inversed_out_train[:, in_loc:in_loc + 1]
    train_q = inversed_out_train[:, in_loc + 1:in_loc + 2]
    train_s1 = inversed_out_train[:, -1].reshape(-1, 1)
    test_p = inversed_out_test[:, in_loc:in_loc + 1]
    test_q = inversed_out_test[:, in_loc + 1:in_loc + 2]
    test_s1 = inversed_out_test[:, -1].reshape(-1, 1)

    train_mae_p, train_mape_p, train_r2_p = calculate_metrics(y_act_train[:, :1], train_p)
    train_mae_q, train_mape_q, train_r2_q = calculate_metrics(y_act_train[:, 1:2], train_q)
    train_mae_s1, train_mape_s1, train_r2_s1 = calculate_metrics(y_act_train[:, 2:], train_s1)
    test_mae_p, test_mape_p, test_r2_p = calculate_metrics(y_act_test[:, :1], test_p)
    test_mae_q, test_mape_q, test_r2_q = calculate_metrics(y_act_test[:, 1:2], test_q)
    test_mae_s1, test_mape_s1, test_r2_s1 = calculate_metrics(y_act_test[:, 2:], test_s1)

    metrics = np.array(
        [
            [train_mae_p, train_mape_p, train_r2_p],
            [train_mae_q, train_mape_q, train_r2_q],
            [train_mae_s1, train_mape_s1, train_r2_s1],
            [test_mae_p, test_mape_p, test_r2_p],
            [test_mae_q, test_mape_q, test_r2_q],
            [test_mae_s1, test_mape_s1, test_r2_s1],
        ],
        dtype=float,
    )

    print(f"Train:p  MAE = {train_mae_p}, MAPE = {train_mape_p}, R2 = {train_r2_p}")
    print(f"Train:q  MAE = {train_mae_q}, MAPE = {train_mape_q}, R2 = {train_r2_q}")
    print(f"Train:s1 MAE = {train_mae_s1}, MAPE = {train_mape_s1}, R2 = {train_r2_s1}")
    print(f"Test:p   MAE = {test_mae_p}, MAPE = {test_mape_p}, R2 = {test_r2_p}")
    print(f"Test:q   MAE = {test_mae_q}, MAPE = {test_mape_q}, R2 = {test_r2_q}")
    print(f"Test:s1  MAE = {test_mae_s1}, MAPE = {test_mape_s1}, R2 = {test_r2_s1}")

    # Plots
    # UT1 q-s1
    x1 = Data_test_real[:row1, s1_loc:]
    x2 = inversed_out_test[:row1, s1_loc:]
    y1 = Data_test_real[:row1, q_loc:q_loc + 1]
    y2 = inversed_out_test[:row1, q_loc:q_loc + 1]
    figure(x1, pick_prediction_x(x1, x2), y1, y2, title="UT1: q-s1")

    # UT1 q-p
    x1 = Data_test_real[:row1, p_loc:p_loc + 1]
    x2 = inversed_out_test[:row1, p_loc:p_loc + 1]
    y1 = Data_test_real[:row1, q_loc:q_loc + 1]
    y2 = inversed_out_test[:row1, q_loc:q_loc + 1]
    figure(x1, pick_prediction_x(x1, x2), y1, y2, title="UT1: q-p")

    # UT5 q-s1
    x1 = Data_test_real[row1:row2, s1_loc:]
    x2 = inversed_out_test[row1:row2, s1_loc:]
    y1 = Data_test_real[row1:row2, q_loc:q_loc + 1]
    y2 = inversed_out_test[row1:row2, q_loc:q_loc + 1]
    figure(x1, pick_prediction_x(x1, x2), y1, y2, title="UT5: q-s1")

    # UT5 q-p
    x1 = Data_test_real[row1:row2, p_loc:p_loc + 1]
    x2 = inversed_out_test[row1:row2, p_loc:p_loc + 1]
    y1 = Data_test_real[row1:row2, q_loc:q_loc + 1]
    y2 = inversed_out_test[row1:row2, q_loc:q_loc + 1]
    figure(x1, pick_prediction_x(x1, x2), y1, y2, title="UT5: q-p")

    # UT7 q-s1
    x1 = Data_test_real[row2:, s1_loc:]
    x2 = inversed_out_test[row2:, s1_loc:]
    y1 = Data_test_real[row2:, q_loc:q_loc + 1]
    y2 = inversed_out_test[row2:, q_loc:q_loc + 1]
    figure(x1, pick_prediction_x(x1, x2), y1, y2, title="UT7: q-s1")

    # UT7 q-p
    x1 = Data_test_real[row2:, p_loc:p_loc + 1]
    x2 = inversed_out_test[row2:, p_loc:p_loc + 1]
    y1 = Data_test_real[row2:, q_loc:q_loc + 1]
    y2 = inversed_out_test[row2:, q_loc:q_loc + 1]
    figure(x1, pick_prediction_x(x1, x2), y1, y2, title="UT7: q-p")

    plot_time_series_diagnostics(Data_test_real, inversed_out_test, row1, row2, "UT5")

    # Loss curve
    x = np.arange(1, len(Loss_train) + 1)
    plt.figure(figsize=(10, 5))
    plt.semilogy(x, Loss_train, label="Train loss", marker="o")
    plt.semilogy(x, Loss_test, label="Test loss", marker="x")
    plt.ylim([1e-6, 1e-1])
    plt.legend()
    plt.grid()
    plt.show()

    # Optional attention map example: uncomment after training to export maps.
    # A_ut5 = extract_interval_mean_attention(
    #     model, Data_test, interval=(row1, row2), length=max_len, stride=window_stride,
    #     batch_size_=batch_size, device_=device, attn_key="spe_attn"
    # )
    # plot_attention_map(A_ut5, "UT5 interval-averaged SPE attention")

    # Save outputs
    Prediction_train = pd.DataFrame(inversed_out_train.tolist())
    Prediction_test = pd.DataFrame(inversed_out_test.tolist())
    Loss_df = pd.DataFrame(Loss.tolist(), columns=["Train loss", "Test loss"])
    Metrics = pd.DataFrame(
        metrics.tolist(),
        index=["train_p", "train_q", "train_s1", "test_p", "test_q", "test_s1"],
        columns=["MAE", "MAPE", "R2"],
    )

    Prediction_test.to_excel("Len-10-SPE-PostSoftmax-Prediction-Test.xlsx", index=False)

    # Export all-point attention tables for the full testing dataset.
    # Output workbook has sheets: "All_Test", "UT1", "TCUI-4", and "TCUI-21".
    export_all_test_attention_tables(
        model=model,
        data_scaled=Data_test,
        row1=row1,
        row2=row2,
        length=max_len,
        device_=device,
        output_file="0503-SPE-v5-PostSoftmax-AllPoint-Attention-Testing-T0T9.xlsx",
        attn_key="spe_attn",
        n_lags=10,
        batch_size_=batch_size,
    )

    # Export all-point B_SPE correction tables for the full testing dataset.
    # Output workbook has sheets: "All_Test", "UT1", "TCUI-4", and "TCUI-21".
    # The model must save B_SPE in get_last_attention_maps(), preferably with key "spe_bias".
    # If both original softmax attention and final SPE-attention are saved, B_SPE can also
    # be computed automatically as their difference.
    export_all_test_bspe_tables(
        model=model,
        data_scaled=Data_test,
        row1=row1,
        row2=row2,
        length=max_len,
        device_=device,
        output_file="0503-SPE-v5-AllPoint-BSPE-Testing-T0T9.xlsx",
        bspe_key="spe_bias",
        spe_attn_key="spe_attn",
        n_lags=10,
        batch_size_=batch_size,
    )

    # Prediction_train.to_excel("SPE-v2_Prediction_Train.xlsx", index=False)
    # Loss_df.to_excel("SPE-v2_Loss.xlsx", index=True)
    # Metrics.to_excel("SPE-v2_Metrics.xlsx")
