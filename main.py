

import os
'''
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"
'''
import random
import copy
import math
import logging
from pathlib import Path
from itertools import combinations
from __future__ import annotations
import re
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from torch_geometric.nn import GCNConv
import optuna
optuna.logging.set_verbosity(optuna.logging.INFO)
from optuna.trial import TrialState
from optuna.importance import get_param_importances
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
import json

import optuna
from optuna.importance import get_param_importances
from optuna.trial import TrialState
from optuna.visualization import plot_parallel_coordinate
import optuna
from optuna.importance import get_param_importances
from optuna.trial import TrialState
import plotly.graph_objects as go

import scikit_posthocs as sp
from scipy.stats import friedmanchisquare
from pathlib import Path
from scipy.stats import studentized_range
from autorank import autorank, plot_stats
from autorank import autorank, plot_stats, create_report



# -------------------- Torch threading --------------------
torch.set_num_threads(6)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# Reproducibility
# =========================================================
def set_global_seed(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    def _seed_worker(worker_id: int):
        worker_seed = seed + int(worker_id)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _seed_worker


SEED = 123
seed_worker = set_global_seed(SEED)

# ============================================================
# Horizons & Models
# ============================================================
HORIZONS = [7]
MODELS_TO_TUNE = ["ASTGCN", "DCRNN", "GCN_FC", "STGCN_GRU", "FC_LSTM", "TCN", "CONVGRU", "CONVLSTM", "TransformerModel"]

# =========================================================
# Data load
# =========================================================
raw_dir = Path(r"D:/doctorate/DATA/clean data/Regression_based_imputation/2000-2024")

station_coords = {
    "kayseri_bolge": (38.687, 35.5, 1099.4136),
    "develi": (38.37441667, 35.47969444, 1201.8264),
    "sariz": (38.47813889, 36.50352778, 1590.1416),
    "tomarza": (38.45219444, 35.79116667, 1397.508),
    "pinarbasi": (38.72513889, 36.39036111, 1534.9728),
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(raw_dir, coords):
    dfs = []
    for f in raw_dir.glob("*.csv"):
        sid = f.stem.lower()
        lat, lon, ele = coords[sid]
        d = pd.read_csv(f, parse_dates=["date"])
        d["station_id"] = sid
        d["latitude"] = lat
        d["longitude"] = lon
        d["elev"] = ele
        # Add date-related columns
        d['year'] = d['date'].dt.year
        d['month'] = d['date'].dt.month
        d['dayofyear'] = d['date'].dt.dayofyear
        
        # --- NEW CODE FOR SIN/COS ENCODING ---
        days_in_year = 365.25 # Use 365.25 to account for leap years
        d['doy_sin'] = np.sin(2 * np.pi * d['dayofyear'] / days_in_year)
        d['doy_cos'] = np.cos(2 * np.pi * d['dayofyear'] / days_in_year)
        # --------------------------------------
        
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)

df_all = load_data(raw_dir, station_coords)

df_all.isna().sum()


# ====== Haversine distance ======
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

# ====== Convert adjacency matrix to edge index and weights ======
def edge_index_weight(A, k=5, symmetric=True):
    np.fill_diagonal(A, 0)
    rows, cols, vals = [], [], []
    for i, row in enumerate(A):
        idx = np.argsort(row)[::-1][:k]  # top k neighbors
        for j in idx:
            if row[j] > 1e-6:
                rows.append(i)
                cols.append(j)
                vals.append(row[j])
    if symmetric:
        rows = rows + cols
        cols = cols + rows[:len(cols)]
        vals = vals + vals
    return (torch.tensor([rows, cols], dtype=torch.long, device=DEVICE),
            torch.tensor(vals, dtype=torch.float32, device=DEVICE))


# ====== Ensure station data format ======
def _ensure_stations_df(stations):
    """Ensure input is a DataFrame with ['latitude', 'longitude', 'elev'] columns."""
    if isinstance(stations, dict):
        return pd.DataFrame.from_dict(stations, orient='index', columns=['latitude', 'longitude', 'elev'])
    if isinstance(stations, pd.DataFrame):
        return stations.copy()
    raise TypeError("stations must be a dict or pandas.DataFrame")

# ====== Build adjacency matrix using Haversine only ======
def build_adj_haversine(stations, k=5, sigma_scale=1.0):
    stations_df = _ensure_stations_df(stations)
    coords = stations_df[['latitude', 'longitude']].values
    n = len(coords)
    A = np.zeros((n, n))

    for i, j in combinations(range(n), 2):
        lat1, lon1 = coords[i]
        lat2, lon2 = coords[j]
        d_h = haversine(lat1, lon1, lat2, lon2)  # horizontal distance in km
        A[i, j] = A[j, i] = np.exp(-d_h / (sigma_scale * 100))

    np.fill_diagonal(A, 0.0)
    return A, stations_df

k_neighbors = 3
A, stations = build_adj_haversine(station_coords, k=k_neighbors)
edge_index, edge_weight = edge_index_weight(A, k=k_neighbors, symmetric=True)

adj_dict = {
    "haversine": {
        "adj_matrix": A,
        "edge_index": edge_index,
        "edge_weight": edge_weight
    }
}


# ----------------------------
# Extreme events labels
# ----------------------------
def extreme_events_labels(
        df_all,
        ref_start=2000, ref_end=2024,
        hw_q=0.90, cw_q=0.10,
        minlen=3,
        hw_season=(5, 9),
        cw_season=(11, 3)):

    df = df_all.copy()
    df["date"] = pd.to_datetime(df["date"])
    
    # Build unique daily index
    unique_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")

    df = df.set_index("date")
    df["month"] = df.index.month
    df["year"]  = df.index.year
    df["month_day"] = df.index.strftime("%m-%d")

    HW_dict = {}
    CW_dict = {}

    def detect_spells(arr, minlen):
        arr = arr.astype(int)
        out = np.zeros(len(arr), dtype=int)
        cnt = 0
        for i, v in enumerate(arr):
            if v == 1:
                cnt += 1
            else:
                if cnt >= minlen:
                    out[i-cnt:i] = 1
                cnt = 0
        if cnt >= minlen:
            out[len(arr)-cnt:] = 1
        return out

    for sid in df["station_id"].unique():
        g = df[df["station_id"] == sid].copy()

        ref = g[(g["year"] >= ref_start) & (g["year"] <= ref_end)]
        if len(ref) < 30:
            ref = g.copy()

        ref["month_day"] = ref.index.strftime("%m-%d")

        thr_hw, thr_cw = {}, {}

        for day in ref["month_day"].unique():
            try:
                dt = pd.to_datetime("2004-" + day)
                ws = (dt - pd.Timedelta(days=7)).strftime("%m-%d")
                we = (dt + pd.Timedelta(days=7)).strftime("%m-%d")
                mask = ref["month_day"].between(ws, we)
                thr_hw[day] = ref.loc[mask, "temperature_max"].quantile(hw_q)
                thr_cw[day] = ref.loc[mask, "temperature_min"].quantile(cw_q)
            except:
                continue

        g["thr_hw"] = g["month_day"].map(thr_hw).fillna(method="ffill").fillna(method="bfill")
        g["thr_cw"] = g["month_day"].map(thr_cw).fillna(method="ffill").fillna(method="bfill")

        g["HW_raw"] = (g["temperature_max"] > g["thr_hw"]).astype(int)
        g["CW_raw"] = (g["temperature_min"] < g["thr_cw"]).astype(int)

        m = g["month"]
        g["HW_raw"] = np.where((m >= hw_season[0]) & (m <= hw_season[1]), g["HW_raw"], 0)
        g["CW_raw"] = np.where((m >= 11) | (m <= 3), g["CW_raw"], 0)

        g["HW_flag"] = detect_spells(g["HW_raw"].values, minlen)
        g["CW_flag"] = detect_spells(g["CW_raw"].values, minlen)

        # This is the CRITICAL FIX
        HW_dict[sid] = g["HW_flag"].reindex(unique_dates, fill_value=0)
        CW_dict[sid] = g["CW_flag"].reindex(unique_dates, fill_value=0)

    HW_labels = pd.DataFrame(HW_dict, index=unique_dates)
    CW_labels = pd.DataFrame(CW_dict, index=unique_dates)
    return HW_labels, CW_labels

def combine_labels(df, heat_labels, cold_labels):
    all_idx = df['date'].sort_values().unique()
    h = heat_labels.reindex(all_idx).fillna(0).astype(int)
    c = cold_labels.reindex(all_idx).fillna(0).astype(int)
    combined = np.select([h == 1, c == 1], [1, 2], default=0)
    return pd.DataFrame(combined, index=all_idx, columns=h.columns)

heat_labels, cold_labels = extreme_events_labels(
    df_all,
    ref_start=2000,
    ref_end=2024,
    hw_q=0.90,
    cw_q=0.10,
    minlen=3,
    hw_season=(5, 9),
    cw_season=(11, 3)
)



labels = combine_labels(df_all, heat_labels, cold_labels)



X_feats=["temperature_mean","temperature_max","temperature_min","relative_humidity", "longitude", "latitude", "elev","dayofyear" , "doy_sin","doy_cos"]



dates=labels.index
stations=labels.columns.tolist()


X = np.stack([
    df_all.pivot(index="date", columns="station_id", values=f)
    .reindex(dates, fill_value=np.nan).values
    for f in X_feats
], axis=2)

y=labels.values.astype("float32")   # (T,N)
T,N,F=np.squeeze(X).shape

# scale
scaler=StandardScaler().fit(X.reshape(-1,F))
X=scaler.transform(X.reshape(-1,F)).reshape(T,N,F).astype("float32")


# =======================================
# Central config
# =======================================

class Config:
    # ---- data ----
    window = 30
    horizon = 1
    batch_size = 30

    # ---- optimization ----
    lr = 1e-3
    weight_decay = 1e-6
    epochs = 600
    patience = 7

    # ---- model ----
    hidden_dim = 64
    dropout = 0.2

    w_heat = 1.0
    w_cold = 1.0
    dcrnn_K = 2 
    astgcn_K = 2   #
    num_nodes = N  #

    n_classes = 3
    input_dim = None

CFG = Config()
CFG.input_dim = X.shape[-1]


# ============================================================
# Dataset and Splits
# ============================================================

class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, start, end, window, horizon):
        self.X = X
        self.y = y
        self.window = int(window)
        self.horizon = int(horizon)

        self.start = int(start)
        self.end = int(end)

        self.idx = [
            (t, t + self.window, t + self.window + self.horizon)
            for t in range(self.start, self.end - self.window - self.horizon + 1)
        ]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        t0, t1, ty = self.idx[i]
        X_seq = self.X[t0:t1]          # (W, N, F)
        y_seq = self.y[t1:ty]          # (H, N)
        return (
            torch.from_numpy(X_seq).float(),
            torch.from_numpy(y_seq).float(),
            int(t1)
        )



def make_splits(X, y, dates, window, horizon, batch_size, run_seed=123):
    """
    Chronological splits:
      - Train = first 70%
      - Val   = next 15%
      - Test  = last 15%
    """
    T = len(dates)
    dates_dt = pd.to_datetime(dates)

    train_end = int(T * 0.70)
    val_end   = int(T * 0.85)

    max_start = T - (window + horizon)

    train_start = 0
    train_stop  = min(train_end, max_start)

    val_start = train_stop
    val_stop  = min(val_end, max_start)

    test_start = val_stop
    test_stop  = T

    if train_stop <= train_start or val_stop <= val_start or test_stop <= test_start:
        raise ValueError(
            f"Invalid splits: "
            f"train=({train_start},{train_stop}), "
            f"val=({val_start},{val_stop}), "
            f"test=({test_start},{test_stop}), "
            f"T={T}, window={window}, horizon={horizon}"
        )

    splits = {
        "train": (train_start, train_stop),
        "val":   (val_start,   val_stop),
        "test":  (test_start,  test_stop),
    }

    datasets = {
        k: SeqDataset(X, y, start, end, window, horizon)
        for k, (start, end) in splits.items()
    }

    loaders = {
        k: torch.utils.data.DataLoader(
            datasets[k],
            batch_size=batch_size,
            shuffle=(k == "train"),
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(run_seed)
        )
        for k in datasets
    }

    n_classes = 3
    print("\nExtreme Events Balance per Split:")
    for name, (start, end) in splits.items():
        y_slice = y[start:end]
        total = y_slice.size
        counts = np.bincount(y_slice.astype(int).ravel(), minlength=n_classes)
        ratios = counts / total
        print(f"{name.capitalize()}: "
              f"Normal={ratios[0]:.2%}, "
              f"Heatwave={ratios[1]:.2%}, "
              f"Coldwave={ratios[2]:.2%}")

    return datasets, loaders


# ============================================================
# Benchmark models
# ============================================================


def make_batched_edge_index(edge_index: torch.Tensor, num_graphs: int, num_nodes: int):
    """
    Replicate edge_index for num_graphs disconnected graphs.
    edge_index: (2, E)
    returns: (2, E*num_graphs)
    """
    device = edge_index.device
    E = edge_index.size(1)

    offsets = (torch.arange(num_graphs, device=device) * num_nodes).repeat_interleave(E)  # (E*num_graphs,)
    edge_index_rep = edge_index.repeat(1, num_graphs)  # (2, E*num_graphs)
    edge_index_rep = edge_index_rep + offsets.unsqueeze(0)  # broadcast to (2, E*num_graphs)
    return edge_index_rep

def make_batched_edge_weight(edge_weight: torch.Tensor, num_graphs: int):
    """
    Repeat edge_weight for num_graphs graphs.
    edge_weight: (E,)
    returns: (E*num_graphs,)
    """
    return edge_weight.repeat(num_graphs)


class STGCN_GRU(nn.Module):
    """
    Correct PyG batching:
    - Applies spatial GCN over (B*W) disconnected graphs in one call.
    - Then applies GRU temporally per node.
    Input:  x (B, W, N, F)
    Output: (B, N, H, C)
    """
    def __init__(self, in_dim, horizon, h=64, dropout=0.0, n_classes=3, gru_layers=1, use_edge_weight=True):
        super().__init__()
        self.horizon = horizon
        self.n_classes = n_classes
        self.h = h
        self.use_edge_weight = use_edge_weight

        # Spatial
        self.gcn1 = GCNConv(in_dim, h)
        self.gcn2 = GCNConv(h, h)
        self.drop = nn.Dropout(dropout)

        # Temporal
        gru_dropout = dropout if (gru_layers > 1 and dropout > 0) else 0.0
        self.gru = nn.GRU(
            input_size=h,
            hidden_size=h,
            num_layers=gru_layers,
            batch_first=False,   # (W, B*N, h)
            dropout=gru_dropout
        )

        # Classifier
        self.fc = nn.Linear(h, horizon * n_classes)

        # cache for batched edge_index to avoid rebuilding every forward
        self._cache = {}  # key: (num_graphs, num_nodes, edge_index_ptr, device) -> edge_index_bw

    def _get_batched_graph(self, edge_index, edge_weight, num_graphs, num_nodes):
        device = edge_index.device
        key = (num_graphs, num_nodes, edge_index.data_ptr(), device)

        if key not in self._cache:
            self._cache[key] = make_batched_edge_index(edge_index, num_graphs, num_nodes)

        edge_index_bw = self._cache[key]

        edge_weight_bw = None
        if self.use_edge_weight and (edge_weight is not None):
            edge_weight_bw = make_batched_edge_weight(edge_weight, num_graphs)

        return edge_index_bw, edge_weight_bw

    def forward(self, x, edge_index, edge_weight=None):
        """
        x: (B, W, N, F)
        returns: (B, N, H, C)
        """
        B, W, N, Fdim = x.shape

        # -------------------------
        # 1) Spatial GCN over (B*W) graphs
        # -------------------------
        num_graphs = B * W

        # Flatten nodes across all graphs: (B*W*N, F)
        x_nodes = x.reshape(B * W * N, Fdim)

        edge_index_bw, edge_weight_bw = self._get_batched_graph(
            edge_index=edge_index,
            edge_weight=edge_weight,
            num_graphs=num_graphs,
            num_nodes=N
        )

        h = self.gcn1(x_nodes, edge_index_bw, edge_weight_bw)   # (B*W*N, h)
        h = torch.relu(h)   # FIX: avoid F.relu (F might be overwritten)
        h = self.drop(h)

        h = self.gcn2(h, edge_index_bw, edge_weight_bw)         # (B*W*N, h)
        h = torch.relu(h)   # FIX: avoid F.relu (F might be overwritten)
        h = self.drop(h)

        # Reshape back: (B, W, N, h)
        h = h.view(B, W, N, self.h)

        # -------------------------
        # 2) Temporal GRU per node
        # -------------------------
        # (B, W, N, h) -> (W, B, N, h) -> (W, B*N, h)
        h = h.permute(1, 0, 2, 3).contiguous()
        h_gru = h.view(W, B * N, self.h)

        _, hn = self.gru(h_gru)            # (layers, B*N, h)
        h_last = hn[-1].view(B, N, self.h) # (B, N, h)

        # -------------------------
        # 3) Classifier
        # -------------------------
        out = self.fc(h_last).view(B, N, self.horizon, self.n_classes)
        return out

    
class FC_LSTM(nn.Module):
    def __init__(self, in_dim, horizon, h=64, dropout=0.0, n_classes=3, num_layers=1):
        super().__init__()
        self.horizon = horizon
        self.n_classes = n_classes
        self.h = h

        lstm_dropout = dropout if (num_layers > 1 and dropout > 0) else 0.0
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=h,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(h, horizon * n_classes)

    def forward(self, x, edge_index=None, edge_weight=None):
        B, W, N, F = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(B * N, W, F)
        _, (hn, _) = self.lstm(x)          # (num_layers, B*N, h)
        h_last = self.drop(hn[-1])         # (B*N, h)
        out = self.fc(h_last).view(B, N, self.horizon, self.n_classes)
        return out




class TCN(nn.Module):
    def __init__(self, in_dim, horizon, h=64, dropout=0.0, n_classes=3, num_layers=2):
        super().__init__()
        assert num_layers >= 1
        self.horizon = horizon
        self.n_classes = n_classes
        self.h = h

        layers = []
        # first conv: in_dim -> h
        layers.append(nn.Conv1d(in_dim, h, kernel_size=3, padding=1))
        # remaining convs: h -> h
        for _ in range(num_layers - 1):
            layers.append(nn.Conv1d(h, h, kernel_size=3, padding=1))
        self.convs = nn.ModuleList(layers)

        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(h, horizon * n_classes)

    def forward(self, x, edge_index=None, edge_weight=None):
        B, W, N, F = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(B * N, W, F)
        x = x.transpose(1, 2)  # (B*N, F, W)

        h = x
        for conv in self.convs:
            h = torch.relu(conv(h))
            h = self.drop(h)

        h = h.mean(dim=2)  # (B*N, h)
        out = self.fc(h).view(B, N, self.horizon, self.n_classes)
        return out


class ConvGRU(nn.Module):
    def __init__(self, in_dim, horizon, h=64, dropout=0.0, n_classes=3, num_layers=1):
        super().__init__()
        self.horizon = horizon
        self.n_classes = n_classes
        self.h = h

        gru_dropout = dropout if (num_layers > 1 and dropout > 0) else 0.0
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=h,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(h, horizon * n_classes)

    def forward(self, x, edge_index=None, edge_weight=None):
        B, W, N, F = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(B * N, W, F)
        _, hn = self.gru(x)                # (num_layers, B*N, h)
        h_last = self.drop(hn[-1])         # (B*N, h)
        out = self.fc(h_last).view(B, N, self.horizon, self.n_classes)
        return out


class ConvLSTM(nn.Module):
    def __init__(self, in_dim, horizon, h=64, dropout=0.0, n_classes=3, num_layers=1):
        super().__init__()
        self.horizon = horizon
        self.n_classes = n_classes
        self.h = h

        lstm_dropout = dropout if (num_layers > 1 and dropout > 0) else 0.0
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=h,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(h, horizon * n_classes)

    def forward(self, x, edge_index=None, edge_weight=None):
        B, W, N, F = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(B * N, W, F)
        _, (hn, _) = self.lstm(x)          # (num_layers, B*N, h)
        h_last = self.drop(hn[-1])         # (B*N, h)
        out = self.fc(h_last).view(B, N, self.horizon, self.n_classes)
        return out


class GCN_FC(nn.Module):

    def __init__(self, in_dim, horizon, h=64, dropout=0.0, n_classes=3):
        super().__init__()
        self.horizon = horizon
        self.n_classes = n_classes
        self.gcn = GCNConv(in_dim, h)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(h, horizon * n_classes)

    def forward(self, x, edge_index, edge_weight=None):
        B, W, N, F = x.shape
        x_last = x[:, -1]  # (B, N, F)
        h = torch.relu(self.gcn(x_last, edge_index, edge_weight))  # (B, N, h)
        h = self.drop(h)
        out = self.fc(h)   # (B, N, H*C)

        return out.view(B, N, self.horizon, self.n_classes)
    
class TransformerModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        horizon: int,
        h: int = 64,
        dropout: float = 0.0,
        n_classes: int = 3,
        n_heads: int = 4,
        num_layers: int = 1,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.n_classes = int(n_classes)

        h = int(h)
        n_heads = int(n_heads)
        num_layers = int(num_layers)

        self.embed = nn.Linear(int(in_dim), h)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=n_heads,
            dropout=float(dropout),
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.drop = nn.Dropout(float(dropout))
        self.fc = nn.Linear(h, int(horizon) * int(n_classes))

    def forward(self, x, edge_index=None, edge_weight=None):
        # x: (B, W, N, F)
        B, W, N, F = x.shape
        x = x.mean(dim=2)             # (B, W, F) node-averaged (your design)
        h = self.embed(x)             # (B, W, h)
        h = self.encoder(h)           # (B, W, h)
        h_last = self.drop(h[:, -1])  # (B, h)
        out = self.fc(h_last)         # (B, H*C)

        out = out.view(B, 1, self.horizon, self.n_classes)
        return out.expand(B, N, self.horizon, self.n_classes).contiguous()


# ============================================================
# Dense graph utilities
# ============================================================

def _edge_to_dense_adj(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    num_nodes: int,
    device: torch.device,
    add_self_loops: bool = True,
    self_loop_weight: float = 1.0
) -> torch.Tensor:
    """
    Build dense adjacency A (N,N) from edge_index/edge_weight.
    - If edge_weight is None -> unweighted adjacency (1s).
    - Self-loops are ADDED (diagonal += self_loop_weight) to avoid overwriting.
    """
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    src = edge_index[0].long()
    dst = edge_index[1].long()

    if edge_weight is None:
        A[src, dst] = 1.0
    else:
        A[src, dst] = edge_weight.float()

    if add_self_loops:
        # SAFE FIX: add to diagonal rather than overwrite
        A.diagonal().add_(float(self_loop_weight))

    return A


def _row_normalize(A: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Row-normalize A: P = D^{-1} A (random-walk transition)."""
    d = A.sum(dim=1, keepdim=True).clamp_min(eps)
    return A / d


def _diffusion_powers(P: torch.Tensor, K: int) -> list[torch.Tensor]:
    """Return [P^0, P^1, ..., P^K]. P^0 is identity."""
    N = P.size(0)
    powers = [torch.eye(N, device=P.device, dtype=P.dtype)]
    if K <= 0:
        return powers
    cur = P
    powers.append(cur)
    for _ in range(2, K + 1):
        cur = cur @ P
        powers.append(cur)
    return powers


def _build_bidirectional_P_powers(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    num_nodes: int,
    device: torch.device,
    K: int,
    add_self_loops: bool = True,
    self_loop_weight: float = 1.0
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Build forward and backward random-walk transition powers:
      Pf = row_norm(A)
      Pb = row_norm(A^T)
    Returns (Pf_powers, Pb_powers), each list of (N,N) matrices.
    """
    A = _edge_to_dense_adj(
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_nodes=num_nodes,
        device=device,
        add_self_loops=add_self_loops,
        self_loop_weight=self_loop_weight
    )
    Pf = _row_normalize(A)
    Pb = _row_normalize(A.transpose(0, 1))
    return _diffusion_powers(Pf, K), _diffusion_powers(Pb, K)


def _build_bidirectional_P_powers_from_denseA(
    A: torch.Tensor,
    K: int
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Same as _build_bidirectional_P_powers, but starts from dense adjacency A (N,N).
    Pf = row_norm(A)
    Pb = row_norm(A^T)
    """
    Pf = _row_normalize(A)
    Pb = _row_normalize(A.transpose(0, 1))
    return _diffusion_powers(Pf, K), _diffusion_powers(Pb, K)


class BiDiffusionConv(nn.Module):
    """
    Bidirectional diffusion convolution:
      out = sum_k (Pf^k X) Wf_k + sum_k (Pb^k X) Wb_k

    Input:
      X: (..., N, Fin)
    Output:
      (..., N, Fout)
    """
    def __init__(self, in_dim: int, out_dim: int, K: int = 2, bias: bool = True):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.K = int(K)

        self.w_f = nn.Parameter(torch.empty(self.K + 1, self.in_dim, self.out_dim))
        self.w_b = nn.Parameter(torch.empty(self.K + 1, self.in_dim, self.out_dim))
        nn.init.xavier_uniform_(self.w_f)
        nn.init.xavier_uniform_(self.w_b)

        self.bias = nn.Parameter(torch.zeros(self.out_dim)) if bias else None

    def forward(
        self,
        X: torch.Tensor,
        Pf_powers: list[torch.Tensor],
        Pb_powers: list[torch.Tensor]
    ) -> torch.Tensor:
        """
        X: (..., N, Fin)
        Pf_powers / Pb_powers: list length (K+1), each (N,N)
        """
        out = 0.0

        for k, Pk in enumerate(Pf_powers):
            Xk = torch.einsum("ij,...jf->...if", Pk, X)
            out = out + torch.einsum("...if,fo->...io", Xk, self.w_f[k])

        for k, Pk in enumerate(Pb_powers):
            Xk = torch.einsum("ij,...jf->...if", Pk, X)
            out = out + torch.einsum("...if,fo->...io", Xk, self.w_b[k])

        if self.bias is not None:
            out = out + self.bias
        return out

# ============================================================
# DCRNN: Diffusion Convolutional Recurrent Neural Network
# ============================================================


class DCRNN(nn.Module):

    def __init__(
        self,
        in_dim: int,
        horizon: int,
        h: int = 64,
        dropout: float = 0.0,
        n_classes: int = 3,
        gru_layers: int = 1,
        K: int = 2,
        use_edge_weight: bool = True,
        add_self_loops: bool = True,
        self_loop_weight: float = 1.0
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.n_classes = int(n_classes)
        self.h = int(h)
        self.K = int(K)
        self.use_edge_weight = bool(use_edge_weight)
        self.add_self_loops = bool(add_self_loops)
        self.self_loop_weight = float(self_loop_weight)

        self.biconv1 = BiDiffusionConv(in_dim, self.h, K=self.K)
        self.biconv2 = BiDiffusionConv(self.h, self.h, K=self.K)
        self.drop = nn.Dropout(dropout)

        gru_dropout = dropout if (gru_layers > 1 and dropout > 0) else 0.0
        self.gru = nn.GRU(
            input_size=self.h,
            hidden_size=self.h,
            num_layers=int(gru_layers),
            batch_first=False,   # (W, B*N, h)
            dropout=gru_dropout
        )

        self.fc = nn.Linear(self.h, self.horizon * self.n_classes)

        # Cache forward/backward diffusion powers per graph instance
        # Key includes: N, edge_index ptr, edge_weight ptr (if used), device, K, self-loop config
        self._cache: dict[tuple, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}

    def _get_bidir_powers(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        N: int,
        device: torch.device
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        ew = edge_weight if (self.use_edge_weight and edge_weight is not None) else None
        key = (
            N,
            int(edge_index.data_ptr()),
            int(ew.data_ptr()) if ew is not None else -1,
            str(device),
            int(self.K),
            bool(self.add_self_loops),
            float(self.self_loop_weight)
        )
        if key not in self._cache:
            Pf_powers, Pb_powers = _build_bidirectional_P_powers(
                edge_index=edge_index,
                edge_weight=ew,
                num_nodes=N,
                device=device,
                K=self.K,
                add_self_loops=self.add_self_loops,
                self_loop_weight=self.self_loop_weight
            )
            self._cache[key] = (Pf_powers, Pb_powers)
        return self._cache[key]

    def forward(self, x, edge_index, edge_weight=None):
        """
        x: (B, W, N, F)
        returns: (B, N, H, C)
        """
        B, W, N, Fin = x.shape
        device = x.device

        Pf_powers, Pb_powers = self._get_bidir_powers(
            edge_index=edge_index,
            edge_weight=edge_weight,
            N=N,
            device=device
        )

        # Bidirectional diffusion per time step (vectorized over B,W)
        h_seq = self.biconv1(x, Pf_powers, Pb_powers)    # (B, W, N, h)
        h_seq = torch.relu(h_seq)
        h_seq = self.drop(h_seq)

        h_seq = self.biconv2(h_seq, Pf_powers, Pb_powers)  # (B, W, N, h)
        h_seq = torch.relu(h_seq)
        h_seq = self.drop(h_seq)

        # GRU over time per node: (B,W,N,h) -> (W, B*N, h)
        h_seq = h_seq.permute(1, 0, 2, 3).contiguous().view(W, B * N, self.h)
        _, hn = self.gru(h_seq)                      # (layers, B*N, h)
        h_last = hn[-1].view(B, N, self.h)          # (B, N, h)

        out = self.fc(h_last).view(B, N, self.horizon, self.n_classes)
        return out


# ============================================================
# ASTGCN: Attention-based Spatio-Temporal Graph Convolution
# ============================================================

class TemporalAttentionNoCollapse(nn.Module):

    def __init__(self, window: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.W = int(window)
        self.d = int(d_model)
        self.drop = nn.Dropout(dropout)

        self.T = nn.Parameter(torch.eye(self.W))

        self.q = nn.Linear(self.d, self.d, bias=False)
        self.k = nn.Linear(self.d, self.d, bias=False)
        self.v = nn.Linear(self.d, self.d, bias=False)

        self.scale = (self.d ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, W, N, D)
        return: (B, W, N, D)
        """
        if x.ndim != 4:
            raise ValueError(f"Expected x.ndim=4, got {x.ndim}")

        B, W, N, D = x.shape
        if D != self.d:
            raise ValueError(f"d_model mismatch: expected {self.d}, got {D}")

        # per-node attention: (B, N, W, D)
        x_n = x.permute(0, 2, 1, 3).contiguous()

        Q = self.q(x_n)  # (B, N, W, D)
        K = self.k(x_n)  # (B, N, W, D)
        V = self.v(x_n)  # (B, N, W, D)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, N, W, W)

        if W == self.W:
            attn = attn + self.T.unsqueeze(0).unsqueeze(0)        # (B, N, W, W)

        attn = torch.softmax(attn, dim=-1)
        attn = self.drop(attn)

        out = torch.matmul(attn, V)                               # (B, N, W, D)
        out = out.permute(0, 2, 1, 3).contiguous()                # (B, W, N, D)
        return out

class SpatialAttention(nn.Module):
    """
    Spatial attention over N:
      input:  x (B, W, N, D)
      output: A_sp (B, N, N)
    """
    def __init__(self, num_nodes: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.N = int(num_nodes)
        self.d = int(d_model)
        self.drop = nn.Dropout(dropout)

        self.S = nn.Parameter(torch.eye(self.N))

        self.q = nn.Linear(self.d, self.d, bias=False)
        self.k = nn.Linear(self.d, self.d, bias=False)

        self.scale = (self.d ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, W, N, D)
        returns: (B, N, N)
        """
        if x.ndim != 4:
            raise ValueError(f"Expected x.ndim=4, got {x.ndim}")

        B, W, N, D = x.shape
        if D != self.d:
            raise ValueError(f"d_model mismatch: expected {self.d}, got {D}")

        Smix = self.S if (N == self.N) else None

        xs = x.mean(dim=1)                     # (B, N, D)
        Q = self.q(xs)                         # (B, N, D)
        K = self.k(xs)                         # (B, N, D)

        attn = torch.matmul(Q, K.transpose(1, 2)) / self.scale  # (B, N, N)
        if Smix is not None:
            attn = attn + Smix.unsqueeze(0)

        attn = torch.softmax(attn, dim=-1)
        attn = self.drop(attn)
        return attn

class ASTGCN(nn.Module):
    """
    Input:  x (B, W, N, F)
    Graph:  edge_index (2,E), edge_weight (E,) optional
    Output: (B, N, H, C)
    """
    def __init__(
        self,
        in_dim: int,
        horizon: int,
        window: int,
        num_nodes: int,
        h: int = 64,
        dropout: float = 0.0,
        n_classes: int = 3,
        num_layers: int = 1,
        K: int = 2,
        use_edge_weight: bool = True,
        add_self_loops: bool = True,
        self_loop_weight: float = 1.0
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.n_classes = int(n_classes)
        self.h = int(h)

        self.window = int(window)
        self.num_nodes = int(num_nodes)
        self.num_layers = int(num_layers)
        self.K = int(K)

        self.use_edge_weight = bool(use_edge_weight)
        self.add_self_loops = bool(add_self_loops)
        self.self_loop_weight = float(self_loop_weight)

        # feature projection
        self.lin_in = nn.Linear(int(in_dim), self.h)

        # attention
        self.t_attn = TemporalAttentionNoCollapse(window=self.window, d_model=self.h, dropout=dropout)
        self.s_attn = SpatialAttention(num_nodes=self.num_nodes, d_model=self.h, dropout=dropout)

        # bidirectional diffusion propagation
        self.biconv = BiDiffusionConv(self.h, self.h, K=self.K)
        self.drop = nn.Dropout(dropout)

        # temporal model
        gru_dropout = dropout if (self.num_layers > 1 and dropout > 0) else 0.0
        self.gru = nn.GRU(
            input_size=self.h,
            hidden_size=self.h,
            num_layers=self.num_layers,
            batch_first=False,     # (W, B*N, h)
            dropout=gru_dropout
        )

        self.fc = nn.Linear(self.h, self.horizon * self.n_classes)

        # Cache base dense adjacency A (not P powers) keyed by graph identity
        self._A_cache: dict[tuple, torch.Tensor] = {}

    def _get_base_A(self, edge_index: torch.Tensor, edge_weight: torch.Tensor | None, device: torch.device) -> torch.Tensor:
        ew = edge_weight if (self.use_edge_weight and edge_weight is not None) else None
        key = (
            self.num_nodes,
            int(edge_index.data_ptr()),
            int(ew.data_ptr()) if ew is not None else -1,
            str(device),
            bool(self.add_self_loops),
            float(self.self_loop_weight)
        )
        if key not in self._A_cache:
            A = _edge_to_dense_adj(
                edge_index=edge_index,
                edge_weight=ew,
                num_nodes=self.num_nodes,
                device=device,
                add_self_loops=self.add_self_loops,
                self_loop_weight=self.self_loop_weight
            )
            self._A_cache[key] = A
        return self._A_cache[key]

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, W, N, F)
        returns: (B, N, H, C)
        """
        if x.ndim != 4:
            raise ValueError(f"Expected x.ndim=4, got {x.ndim}")

        B, W, N, Fin = x.shape
        device = x.device

        if N != self.num_nodes:
            raise ValueError(f"ASTGCN initialized for num_nodes={self.num_nodes} but got N={N}.")

        # 1) project
        h = self.lin_in(x)                     # (B, W, N, h)

        # 2) temporal attention per node
        h_t = self.t_attn(h)                   # (B, W, N, h)

        # 3) spatial attention (B, N, N)
        A_sp = self.s_attn(h_t)                # (B, N, N)

        # 4) base dense adjacency (N,N)
        A_base = self._get_base_A(edge_index=edge_index, edge_weight=edge_weight, device=device)  # (N,N)

        # 5) attention-modulated bidirectional diffusion per batch item
        # Build per-batch powers and propagate (N is small in your study, OK).
        h_out = []
        for b in range(B):
            # elementwise modulation
            A_attn = A_base * A_sp[b]                          # (N,N)

            # ensure self-loops remain present (optional but recommended)
            if self.add_self_loops:
                A_attn.diagonal().add_(0.0)  # no-op; adjacency already had loops; keep for clarity

            Pf_powers_b, Pb_powers_b = _build_bidirectional_P_powers_from_denseA(A_attn, K=self.K)

            # propagate each timestep sequence (W,N,h)
            hb = self.biconv(h_t[b], Pf_powers_b, Pb_powers_b)  # (W, N, h)
            hb = torch.relu(hb)
            hb = self.drop(hb)
            h_out.append(hb)

        h_out = torch.stack(h_out, dim=0)      # (B, W, N, h)

        # 6) GRU over time per node
        h_seq = h_out.permute(1, 0, 2, 3).contiguous().view(W, B * N, self.h)  # (W,B*N,h)
        _, hn = self.gru(h_seq)                # (layers, B*N, h)
        h_last = hn[-1].view(B, N, self.h)     # (B, N, h)

        out = self.fc(h_last).view(B, N, self.horizon, self.n_classes)
        return out




def build_model(
    model_name: str,
    in_dim: int,
    horizon: int,
    h: int,
    dropout: float,
    n_classes: int,
    num_layers: int,
    CFG
):
    model_name_u = str(model_name).upper()

    if model_name_u == "GCN_FC":
        return GCN_FC(in_dim=in_dim, horizon=horizon, h=h, dropout=dropout, n_classes=n_classes)

    if model_name_u == "STGCN_GRU":
        return STGCN_GRU(in_dim=in_dim, horizon=horizon, h=h, dropout=dropout,
                         n_classes=n_classes, gru_layers=num_layers)

    if model_name_u == "FC_LSTM":
        return FC_LSTM(in_dim=in_dim, horizon=horizon, h=h, dropout=dropout,
                       n_classes=n_classes, num_layers=num_layers)

    if model_name_u == "CONVGRU":
        return ConvGRU(in_dim=in_dim, horizon=horizon, h=h, dropout=dropout,
                       n_classes=n_classes, num_layers=num_layers)

    if model_name_u == "CONVLSTM":
        return ConvLSTM(in_dim=in_dim, horizon=horizon, h=h, dropout=dropout,
                        n_classes=n_classes, num_layers=num_layers)

    if model_name_u == "TCN":
        return TCN(in_dim=in_dim, horizon=horizon, h=h, dropout=dropout,
                   n_classes=n_classes, num_layers=num_layers)

    if model_name_u == "TRANSFORMERMODEL": 
        return TransformerModel( 
            in_dim=in_dim,
            horizon=horizon,
            h=h,
            dropout=dropout,
            n_classes=n_classes,
            n_heads=int(getattr(CFG, "n_heads", 4)),
            num_layers=int(num_layers) 
            )


    # ----------------------------
    # DCRNN (DCRNN-inspired)
    # ----------------------------
    if model_name_u == "DCRNN":
        return DCRNN(
            in_dim=in_dim,
            horizon=horizon,
            h=h,
            dropout=dropout,
            n_classes=n_classes,
            gru_layers=num_layers,
            K=int(getattr(CFG, "dcrnn_K", 2)),
            use_edge_weight=True,
            add_self_loops=True,
            self_loop_weight=float(getattr(CFG, "self_loop_weight", 1.0))
        )

    # ----------------------------
    # ASTGCN (attention-modulated bidirectional diffusion)
    # ----------------------------
    if model_name_u == "ASTGCN":
        # Always infer num_nodes from CFG if set, otherwise fall back to global y (as you previously did)
        num_nodes = int(getattr(CFG, "num_nodes", 0))
        if num_nodes <= 0:
            num_nodes = int(y.shape[1]) if "y" in globals() else 0
        if num_nodes <= 0:
            raise ValueError("ASTGCN requires CFG.num_nodes (preferred) or global y to infer num_nodes.")

        # IMPORTANT: window must match the data window used in the loaders for this trial/run
        window = int(getattr(CFG, "window", 0))
        if window <= 0:
            raise ValueError("ASTGCN requires CFG.window to be set to the current trial window.")

        return ASTGCN(
            in_dim=in_dim,
            horizon=horizon,
            window=window,
            num_nodes=num_nodes,
            h=h,
            dropout=dropout,
            n_classes=n_classes,
            num_layers=num_layers,
            K=int(getattr(CFG, "astgcn_K", 2)),
            use_edge_weight=True,
            add_self_loops=True,
            self_loop_weight=float(getattr(CFG, "self_loop_weight", 1.0))
        )

    raise ValueError(f"Unknown model_name: {model_name}")


MODEL_REGISTRY = {
    "DCRNN": DCRNN,
    "ASTGCN": ASTGCN,
    "STGCN_GRU": STGCN_GRU,
    "FC_LSTM": FC_LSTM,
    "TCN": TCN,
    "GCN_FC": GCN_FC,
    "CONVGRU": ConvGRU,
    "CONVLSTM": ConvLSTM,
    "TransformerModel": TransformerModel
}



# ============================================================
# 3)  Run epoch
# ============================================================

def run_epoch(loader, model, loss_fn, edge_index, edge_weight=None,
                       optimizer=None, device="cpu"):
    train = optimizer is not None
    model.train() if train else model.eval()

    losses, y_true, y_pred = [], [], []

    for xb, yb, _t1 in loader:  # ✅ dataset returns 3 items
        xb = xb.to(device)
        yb = yb.to(device).long()  # (B, H, N)

        if train:
            optimizer.zero_grad()

        # ✅ for non-graph models, they ignore edge_index anyway
        logits = model(xb, edge_index, edge_weight)  # (B, N, H, C)

        loss = loss_fn(
            logits.reshape(-1, logits.shape[-1]),
            yb.reshape(-1)
        )

        if train:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        y_true.append(yb.detach().cpu().numpy())
        y_pred.append(logits.detach().cpu().numpy())

    y_true = np.concatenate(y_true).ravel()
    y_pred = np.concatenate(y_pred).reshape(-1, model.n_classes)
    y_pred_cls = y_pred.argmax(axis=1)

    metrics = {
        "acc": accuracy_score(y_true, y_pred_cls),
        "f1": f1_score(y_true, y_pred_cls, average="macro", zero_division=0),
        "prec": precision_score(y_true, y_pred_cls, average="macro", zero_division=0),
        "rec": recall_score(y_true, y_pred_cls, average="macro", zero_division=0),
    }
    return float(np.mean(losses)), metrics



class EarlyStoppingFlexible:
    def __init__(self, monitor="val_f1", mode="max", patience=5,
                 min_delta=0.0, ignore_zero=True, verbose=True):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.ignore_zero = ignore_zero
        self.verbose = verbose

        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_state_dict = None

        if mode not in ["min", "max"]:
            raise ValueError("mode must be 'min' or 'max'")

    def step(self, score, model):
        if self.ignore_zero and score == 0:
            return

        improved = (
            self.best_score is None or
            (score > self.best_score + self.min_delta if self.mode == "max"
             else score < self.best_score - self.min_delta)
        )

        if improved:
            self.best_score = score
            self.best_state_dict = copy.deepcopy(model.state_dict())
            self.counter = 0
            if self.verbose:
                print(f"Improved {self.monitor}: {score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement ({score:.4f}). Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered! Best {self.monitor}: {self.best_score:.4f}")



# ----------------------------
# Class imbalance
# ----------------------------
weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0,1,2]),
    y=y.reshape(-1)
)

class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
print(class_weights)

w_heat_uper = float(class_weights[1].item())
w_cold_uper = float(class_weights[2].item())


# ============================================================
# 3) Model training
# ============================================================
def train_model(model, train_loader, val_loader, CFG,
                         edge_index, edge_weight=None,
                         device="cpu", verbose=1):
    model.to(device)

    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, float(CFG.w_heat), float(CFG.w_cold)], dtype=torch.float32).to(device)
    )


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CFG.lr,
        weight_decay=CFG.weight_decay
    )

    early_stopper = EarlyStoppingFlexible(
        monitor="val_f1",
        mode="max",
        patience=CFG.patience,
        verbose=(verbose == 1)
    )

    for epoch in range(1, CFG.epochs + 1):
        tr_loss, tr_metrics = run_epoch(
            loader=train_loader,
            model=model,
            loss_fn=loss_fn,
            edge_index=edge_index,
            edge_weight=edge_weight,
            optimizer=optimizer,
            device=device
        )

        va_loss, va_metrics = run_epoch(
            loader=val_loader,
            model=model,
            loss_fn=loss_fn,
            edge_index=edge_index,
            edge_weight=edge_weight,
            optimizer=None,
            device=device
        )

        if verbose == 1:
            print(
                f"[Epoch {epoch:03d}] "
                f"Train: loss={tr_loss:.4f}, F1={tr_metrics['f1']:.3f} | "
                f"Val: loss={va_loss:.4f}, F1={va_metrics['f1']:.3f}"
            )

        early_stopper.step(va_metrics["f1"], model)
        if early_stopper.early_stop:
            if verbose == 1:
                print(
                    f"Early stopping at epoch {epoch}. "
                    f"Best Val F1 = {early_stopper.best_score:.3f}"
                )
            model.load_state_dict(early_stopper.best_state_dict)
            break

# ============================================================
# Event-level (spell-based) utilities
# ============================================================

def extract_spells(binary_arr, minlen=3):
    spells = []
    start = None
    for i, v in enumerate(binary_arr):
        if v == 1 and start is None:
            start = i
        elif v == 0 and start is not None:
            if i - start >= minlen:
                spells.append((start, i - 1))
            start = None
    if start is not None and len(binary_arr) - start >= minlen:
        spells.append((start, len(binary_arr) - 1))
    return spells


def count_event_matches(true_spells, pred_spells, min_overlap=1):
    detected = 0
    used_preds = set()
    for ts in true_spells:
        for i, ps in enumerate(pred_spells):
            if i in used_preds:
                continue
            overlap = max(0, min(ts[1], ps[1]) - max(ts[0], ps[0]) + 1)
            if overlap >= min_overlap:
                detected += 1
                used_preds.add(i)
                break
    return detected


def event_level_metrics(
    y_true,
    y_pred,
    stations,
    minlen=3,
    min_overlap=1,
    label_names=("HW", "CW")
):
    """
    y_true, y_pred: (T_days, N, 2) binary arrays
    """
    results = {}
    T, N, C = y_true.shape

    for li, label in enumerate(label_names):
        true_events = pred_events = detected_events = 0

        for si in range(N):
            yt = y_true[:, si, li]
            yp = y_pred[:, si, li]

            true_spells = extract_spells(yt, minlen)
            pred_spells = extract_spells(yp, minlen)

            true_events += len(true_spells)
            pred_events += len(pred_spells)
            detected_events += count_event_matches(
                true_spells, pred_spells, min_overlap
            )

        precision = detected_events / pred_events if pred_events > 0 else 0.0
        recall    = detected_events / true_events if true_events > 0 else 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

        results[label] = {
            "event_precision": precision,
            "event_recall": recall,
            "event_f1": f1,
            "n_true_events": true_events,
            "n_pred_events": pred_events,
            "n_detected_events": detected_events
        }

    return results



# ============================================================
# 4) Evaluation
# ============================================================

def evaluate(
    loader,
    model,
    edge_index,
    edge_weight=None,
    device="cpu",
    minlen=3,
    min_overlap=1,
    eval_start_day: int | None = None,
    eval_end_day: int | None = None,
    y_global: np.ndarray | None = None,
):

    model.eval()

    ds = loader.dataset
    if len(ds.idx) == 0:
        raise ValueError("Dataset is empty: cannot evaluate.")

    if y_global is None:
        # fall back to your global y variable if present
        if "y" not in globals():
            raise ValueError("y_global not provided and global y not found.")
        y_global = y  # expects shape (T, N)

    # Determine evaluation calendar range
    if eval_start_day is None or eval_end_day is None:
        first_t1 = int(ds.idx[0][1])
        last_ty  = int(ds.idx[-1][2])
        start_day = first_t1
        end_day   = last_ty - 1
    else:
        start_day = int(eval_start_day)
        end_day   = int(eval_end_day)
        if end_day < start_day:
            raise ValueError(f"Invalid eval range: start_day={start_day} > end_day={end_day}")

    T_days = end_day - start_day + 1
    N = int(y_global.shape[1])

    prob_sum = None
    prob_cnt = np.zeros((T_days, N), dtype=np.float64)

    # Ground-truth labels: take directly from y_global on the fixed range
    # This ensures identical "True Events" across models for the same eval range.
    if y_global.shape[0] <= end_day:
        raise ValueError(f"y_global length {y_global.shape[0]} is too short for end_day={end_day}")
    y_true_day = y_global[start_day:end_day + 1, :].astype(np.int32, copy=False)  # (T_days, N)

    with torch.no_grad():
        for xb, yb, t1_batch in loader:
            xb = xb.to(device)

            logits = model(xb, edge_index, edge_weight)  # (B, N, H, C)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            B, N2, H, C = probs.shape
            if N2 != N:
                raise ValueError(f"N mismatch: probs has N={N2}, but y_global has N={N}")

            if prob_sum is None:
                prob_sum = np.zeros((T_days, N, C), dtype=np.float64)

            if torch.is_tensor(t1_batch):
                t1_batch = t1_batch.cpu().numpy().astype(int)

            for b in range(B):
                t1 = int(t1_batch[b])
                for h in range(H):
                    day_abs = t1 + h
                    if day_abs < start_day or day_abs > end_day:
                        continue
                    di = day_abs - start_day
                    prob_sum[di] += probs[b, :, h, :]
                    prob_cnt[di] += 1.0

    if prob_sum is None:
        raise ValueError("No probabilities were accumulated. Check eval range vs loader coverage.")

    safe_cnt = np.maximum(prob_cnt, 1.0)
    prob_mean = prob_sum / safe_cnt[:, :, None]
    y_pred_day = prob_mean.argmax(axis=-1).astype(np.int32)

    # ----------------------------
    # Day-level metrics
    # ----------------------------
    y_true_flat = y_true_day.ravel()
    y_pred_flat = y_pred_day.ravel()

    day_metrics = {
        "accuracy": accuracy_score(y_true_flat, y_pred_flat),
        "f1_macro": f1_score(y_true_flat, y_pred_flat, average="macro", zero_division=0),
        "precision_macro": precision_score(y_true_flat, y_pred_flat, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true_flat, y_pred_flat, average="macro", zero_division=0),
        "y_true": y_true_flat,
        "y_pred": y_pred_flat,
    }

    # ----------------------------
    # Event-level metrics
    # ----------------------------
    true_evt = np.zeros((T_days, N, 2), dtype=np.int32)
    pred_evt = np.zeros((T_days, N, 2), dtype=np.int32)

    true_evt[..., 0] = (y_true_day == 1).astype(np.int32)  # HW
    true_evt[..., 1] = (y_true_day == 2).astype(np.int32)  # CW
    pred_evt[..., 0] = (y_pred_day == 1).astype(np.int32)
    pred_evt[..., 1] = (y_pred_day == 2).astype(np.int32)

    event_metrics = event_level_metrics(
        y_true=true_evt,
        y_pred=pred_evt,
        stations=range(N),
        minlen=minlen,
        min_overlap=min_overlap
    )

    return {
        **day_metrics,
        "event_level": event_metrics,
        "y_true_day": y_true_day,
        "y_pred_day": y_pred_day,
        "eval_start_day": start_day,
        "eval_end_day": end_day,
    }


# ============================================================
# Logging
# ============================================================
def setup_logger():
    logger = logging.getLogger("ExperimentLogger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = setup_logger()


# ============================================================
# Experiment runner
# ============================================================

def run_experiments(
    model_name,
    X, y, dates,
    CFG,
    device="cpu",
    n_runs=10
):
    """
    Baseline experiments:
    - Day-level metrics (calendar-aligned via evaluate())
    - Event-level metrics (spell-based) via evaluate()["event_level"]
    - Mean ± std across n_runs
    - Preserves original modularity and call pattern
    """

    logger.info(f"=== Baseline Experiment: {model_name} ===")
    model_class = MODEL_REGISTRY[model_name]

    # -----------------------------
    # Storage: day-level (global)
    # -----------------------------
    overall = {
        "accuracy": [],
        "f1_macro": [],
        "precision_macro": [],
        "recall_macro": []
    }

    # -----------------------------
    # Storage: day-level per-class (0/1/2)
    # -----------------------------
    per_class = {
        "Normal":   {"precision": [], "recall": [], "f1": []},
        "Heatwave": {"precision": [], "recall": [], "f1": []},
        "Coldwave": {"precision": [], "recall": [], "f1": []},
    }

    # -----------------------------
    # Storage: event-level (HW/CW)
    # -----------------------------
    event_level = {
        "Heatwave": {"event_precision": [], "event_recall": [], "event_f1": [],
                     "n_true_events": [], "n_pred_events": [], "n_detected_events": []},
        "Coldwave": {"event_precision": [], "event_recall": [], "event_f1": [],
                     "n_true_events": [], "n_pred_events": [], "n_detected_events": []},
    }

    # -----------------------------
    # Repeated runs
    # -----------------------------
    for run in range(1, n_runs + 1):
        run_seed = SEED + run
        set_global_seed(run_seed)

        _, loaders = make_splits(
            X, y, dates,
            window=CFG.window,
            horizon=CFG.horizon,
            batch_size=CFG.batch_size,
            run_seed=run_seed
        )

        model = model_class(
            in_dim=CFG.input_dim,
            horizon=CFG.horizon,
            h=CFG.hidden_dim,
            dropout=CFG.dropout,
            n_classes=CFG.n_classes
        ).to(device)

        train_model(
            model,
            loaders["train"],
            loaders["val"],
            CFG,
            edge_index=edge_index,
            edge_weight=edge_weight,
            device=device
        )

        # ============================================================
        # Event-aware evaluation (calendar-aligned + event-level spells)
        # ============================================================
        metrics = evaluate(
            loaders["test"],
            model,
            edge_index=edge_index,
            edge_weight=edge_weight,
            device=device,
            minlen=3,
            min_overlap=1
        )

        # ---- global day-level ----
        for k in overall:
            overall[k].append(metrics[k])

        # ---- per-class day-level ----
        # IMPORTANT: use calendar-aligned arrays if present
        if "y_true_day" in metrics and "y_pred_day" in metrics:
            yt = metrics["y_true_day"].ravel()
            yp = metrics["y_pred_day"].ravel()
        else:
            # fallback (should not happen if using event-aware evaluate)
            yt = metrics["y_true"]
            yp = metrics["y_pred"]

        for cls_id, cls_name in zip([0, 1, 2], ["Normal", "Heatwave", "Coldwave"]):
            ytc = (yt == cls_id).astype(int)
            ypc = (yp == cls_id).astype(int)

            per_class[cls_name]["precision"].append(
                precision_score(ytc, ypc, zero_division=0)
            )
            per_class[cls_name]["recall"].append(
                recall_score(ytc, ypc, zero_division=0)
            )
            per_class[cls_name]["f1"].append(
                f1_score(ytc, ypc, zero_division=0)
            )

        # ---- event-level (spell-based) ----
        evt = metrics.get("event_level", None)
        if evt is not None:
            # evt keys: "HW", "CW" (as defined in event_level_metrics)
            if "HW" in evt:
                event_level["Heatwave"]["event_precision"].append(float(evt["HW"]["event_precision"]))
                event_level["Heatwave"]["event_recall"].append(float(evt["HW"]["event_recall"]))
                event_level["Heatwave"]["event_f1"].append(float(evt["HW"]["event_f1"]))
                event_level["Heatwave"]["n_true_events"].append(int(evt["HW"]["n_true_events"]))
                event_level["Heatwave"]["n_pred_events"].append(int(evt["HW"]["n_pred_events"]))
                event_level["Heatwave"]["n_detected_events"].append(int(evt["HW"]["n_detected_events"]))

            if "CW" in evt:
                event_level["Coldwave"]["event_precision"].append(float(evt["CW"]["event_precision"]))
                event_level["Coldwave"]["event_recall"].append(float(evt["CW"]["event_recall"]))
                event_level["Coldwave"]["event_f1"].append(float(evt["CW"]["event_f1"]))
                event_level["Coldwave"]["n_true_events"].append(int(evt["CW"]["n_true_events"]))
                event_level["Coldwave"]["n_pred_events"].append(int(evt["CW"]["n_pred_events"]))
                event_level["Coldwave"]["n_detected_events"].append(int(evt["CW"]["n_detected_events"]))

    # =========================================================
    # SUMMARY TABLES (MEAN ± STD)
    # =========================================================

    # ---- overall (day-level) ----
    overall_df = pd.DataFrame(overall)
    overall_summary = overall_df.agg(["mean", "std"]).T

    # ---- per-class (day-level) ----
    class_summary_df = pd.DataFrame({
        cls: {
            "precision_mean": np.mean(per_class[cls]["precision"]),
            "precision_std":  np.std(per_class[cls]["precision"]),
            "recall_mean":    np.mean(per_class[cls]["recall"]),
            "recall_std":     np.std(per_class[cls]["recall"]),
            "f1_mean":        np.mean(per_class[cls]["f1"]),
            "f1_std":         np.std(per_class[cls]["f1"]),
        }
        for cls in per_class
    }).T

    # ---- event-level (HW/CW) ----
    event_summary_df = pd.DataFrame({
        cls: {
            "event_precision_mean": np.mean(event_level[cls]["event_precision"]) if event_level[cls]["event_precision"] else 0.0,
            "event_precision_std":  np.std(event_level[cls]["event_precision"])  if event_level[cls]["event_precision"] else 0.0,
            "event_recall_mean":    np.mean(event_level[cls]["event_recall"])    if event_level[cls]["event_recall"] else 0.0,
            "event_recall_std":     np.std(event_level[cls]["event_recall"])     if event_level[cls]["event_recall"] else 0.0,
            "event_f1_mean":        np.mean(event_level[cls]["event_f1"])        if event_level[cls]["event_f1"] else 0.0,
            "event_f1_std":         np.std(event_level[cls]["event_f1"])         if event_level[cls]["event_f1"] else 0.0,

            "n_true_events_mean":      np.mean(event_level[cls]["n_true_events"])      if event_level[cls]["n_true_events"] else 0.0,
            "n_true_events_std":       np.std(event_level[cls]["n_true_events"])       if event_level[cls]["n_true_events"] else 0.0,
            "n_pred_events_mean":      np.mean(event_level[cls]["n_pred_events"])      if event_level[cls]["n_pred_events"] else 0.0,
            "n_pred_events_std":       np.std(event_level[cls]["n_pred_events"])       if event_level[cls]["n_pred_events"] else 0.0,
            "n_detected_events_mean":  np.mean(event_level[cls]["n_detected_events"])  if event_level[cls]["n_detected_events"] else 0.0,
            "n_detected_events_std":   np.std(event_level[cls]["n_detected_events"])   if event_level[cls]["n_detected_events"] else 0.0,
        }
        for cls in ["Heatwave", "Coldwave"]
    }).T

    return {
        "summary": overall_summary,          # day-level global metrics (mean ± std)
        "all_runs": overall_df,              # per-run day-level global metrics
        "class_summary": class_summary_df,   # day-level per-class metrics
        "event_summary": event_summary_df    # event-level spell-based metrics
    }


#=======================================================
# Optuna HPO
#=======================================================

OPTUNA_DIR = Path("D:/doctorate/heatwaveprediction/results/optuna_benchmark")
OPTUNA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HPO objective builder for (model, horizon)
# ============================================================

def build_objective(model_name: str, horizon: int, base_seed: int = 123, max_epochs: int = 20):

    model_name_u = str(model_name).upper()

    def objective(trial: optuna.Trial):
        # -----------------------------
        # Baseline hyperparameters (ALL models)
        # -----------------------------
        hidden_dim   = trial.suggest_int("hidden_dim", 8, 128, step=8)
        dropout      = trial.suggest_float("dropout", 0.0, 0.5)
        window       = trial.suggest_categorical("window", [3, 5, 7, 15, 20, 30, 45])
        batch_size   = trial.suggest_int("batch_size", 8, 64)

        lr           = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)

        w_heat       = trial.suggest_float("w_heat", 1.0, float(w_heat_uper))
        w_cold       = trial.suggest_float("w_cold", 1.0, float(w_cold_uper))
        num_layers = trial.suggest_int("num_layers", 1, 3)
       
        # -----------------------------
        # Trial-specific CFG 
        # -----------------------------
        CFG_trial = deepcopy(CFG)
        CFG_trial.window = int(window)
        CFG_trial.horizon = int(horizon)        
        CFG_trial.batch_size = int(batch_size)  
        CFG_trial.hidden_dim = int(hidden_dim)  
        CFG_trial.dropout = float(dropout)      
        CFG_trial.lr = float(lr)
        CFG_trial.weight_decay = float(weight_decay)  
        CFG_trial.w_heat = float(w_heat)        
        CFG_trial.w_cold = float(w_cold) 
        CFG_trial.num_layers = int(num_layers)
        
        class_weights = torch.tensor([1.0, w_heat, w_cold], dtype=torch.float32, device=DEVICE)

        # -----------------------------
        # Data (fresh loaders per trial)
        # -----------------------------
        _, loaders = make_splits(
            X, y, dates,
            window=int(window),
            horizon=int(horizon),
            batch_size=int(batch_size),
            run_seed=int(base_seed)
        )

        # -----------------------------
        # Model (fresh per trial)
        # -----------------------------
        model = build_model(
            model_name=model_name,
            in_dim=int(CFG.input_dim),
            horizon=int(horizon),
            h=int(hidden_dim),
            dropout=float(dropout),
            n_classes=int(CFG.n_classes),
            num_layers=int(num_layers),
            CFG=CFG_trial
        ).to(DEVICE)

        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

        early_stopper = EarlyStoppingFlexible(
            monitor="val_f1",
            mode="max",
            patience=int(getattr(CFG, "patience", 7)),
            min_delta=0.0,
            ignore_zero=True,
            verbose=False
        )

        best_epoch = -1
        best_val_metrics = None

        for epoch in range(int(max_epochs)):
            run_epoch(
                loader=loaders["train"],
                model=model,
                loss_fn=loss_fn,
                edge_index=edge_index,
                edge_weight=edge_weight,
                optimizer=optimizer,
                device=DEVICE
            )

            _, val_metrics = run_epoch(
                loader=loaders["val"],
                model=model,
                loss_fn=loss_fn,
                edge_index=edge_index,
                edge_weight=edge_weight,
                optimizer=None,
                device=DEVICE
            )

            f1 = float(val_metrics["f1"])
            trial.report(f1, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            prev_best = early_stopper.best_score
            early_stopper.step(f1, model)
            if early_stopper.best_score != prev_best:
                best_epoch = epoch
                best_val_metrics = val_metrics

            if early_stopper.early_stop:
                break

        if early_stopper.best_state_dict is not None:
            model.load_state_dict(early_stopper.best_state_dict)

        if best_val_metrics is None:
            _, best_val_metrics = run_epoch(
                loader=loaders["val"],
                model=model,
                loss_fn=loss_fn,
                edge_index=edge_index,
                edge_weight=edge_weight,
                optimizer=None,
                device=DEVICE
            )

        trial.set_user_attr("best_epoch", int(best_epoch))
        trial.set_user_attr("val_acc", float(best_val_metrics["acc"]))
        trial.set_user_attr("val_f1_macro", float(best_val_metrics["f1"]))
        trial.set_user_attr("val_prec_macro", float(best_val_metrics["prec"]))
        trial.set_user_attr("val_rec_macro", float(best_val_metrics["rec"]))

        best_score = float(
            early_stopper.best_score if early_stopper.best_score is not None else best_val_metrics["f1"]
        )

        # cleanup
        del model, optimizer, loss_fn, loaders
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return best_score

    return objective

# ============================================================
# 5) Run Optuna studies for each (model, horizon)
# ============================================================
def run_study(
    model_name: str,
    horizon: int,
    n_trials: int = 100,
    seed: int = 123,
    max_epochs: int = 20
):
    storage_path = OPTUNA_DIR / f"hpo_{model_name}_H{horizon}.db"

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name=f"HPO_{model_name}_H{horizon}",
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True
    )

    obj = build_objective(model_name=model_name, horizon=horizon, base_seed=seed, max_epochs=max_epochs)
    study.optimize(obj, n_trials=n_trials, show_progress_bar=True)
    return study


def _completed_trials(study):
    return [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]


def _trials_to_df(completed):
    rows = []
    for t in completed:
        rows.append({
            "trial": t.number,
            "value_best_val_f1": t.value,
            **t.params,
            "best_epoch": t.user_attrs.get("best_epoch"),
            "val_acc": t.user_attrs.get("val_acc"),
            "val_f1_macro": t.user_attrs.get("val_f1_macro"),
            "val_prec_macro": t.user_attrs.get("val_prec_macro"),
            "val_rec_macro": t.user_attrs.get("val_rec_macro"),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("value_best_val_f1", ascending=False).reset_index(drop=True)

    # fixed-depth negative control (GCN_FC) schema consistency
    if "num_layers" not in df.columns:
        df["num_layers"] = 1
    return df


def execute_hpo_grid(
    models_to_tune,
    horizons,
    n_trials: int = 100,
    seed: int = 123,
    max_epochs: int = 20
):
    studies = {}
    best_params = {}
    trial_metrics = {}

    for model_name in models_to_tune:
        for horizon in horizons:
            print(f"\n==============================")
            print(f" Running HPO: {model_name} | Horizon={horizon}")
            print(f"==============================")

            study = run_study(model_name=model_name, horizon=horizon, n_trials=n_trials, seed=seed, max_epochs=max_epochs)
            studies[(model_name, horizon)] = study

            completed = _completed_trials(study)
            if completed:
                best_params[(model_name, horizon)] = {
                    "best_value": float(study.best_value),
                    "best_params": dict(study.best_params)
                }
            else:
                best_params[(model_name, horizon)] = {"best_value": None, "best_params": {}}

            trial_df = _trials_to_df(completed)
            trial_metrics[(model_name, horizon)] = trial_df

            # Save per-study artifacts
            trial_df.to_csv(OPTUNA_DIR / f"trials_{model_name}_H{horizon}.csv", index=False)
            with open(OPTUNA_DIR / f"best_{model_name}_H{horizon}.json", "w", encoding="utf-8") as f:
                json.dump(best_params[(model_name, horizon)], f, indent=2)

    return studies, best_params, trial_metrics


# ============================================================
# 6) Execute the full benchmark HPO grid
# ============================================================
studies, best_params, trial_metrics = execute_hpo_grid(
    MODELS_TO_TUNE,
    HORIZONS,
    n_trials=100,
    seed=123,
    max_epochs=30
)



# ============================================================
# FINAL retraining: (model × horizon × graph)
# ============================================================


OPTUNA_DIR = Path(r"D:/doctorate/heatwaveprediction/results/optuna_benchmark")
OUT_DIR = OPTUNA_DIR / "final_retraining"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_DIR = OUT_DIR / "final_weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

FINAL_CSV = OUT_DIR / "final_benchmark_results.csv"



# =========================
# CONFIG
# =========================
@dataclass
class FinalRetrainConfig:
    default_epochs: int = 20            # fallback if best_epoch missing
    best_epoch_is_zero_based: bool = True  # your objective stores epoch starting at 0
    num_workers: int = 0               # keep 0 to avoid Windows multiprocessing issues
    pin_memory: bool = True
    deterministic: bool = True
    save_logits: bool = False          # keep False for speed/storage


# =========================
# STUDY LOADING
# =========================
_DB_RE = re.compile(r"^hpo_(?P<model>.+)_H(?P<horizon>\d+)\.db$", re.IGNORECASE)

def iter_hpo_dbs(
    optuna_dir: Path,
    models_to_run: list[str] | None = None,
    horizons_to_run: list[int] | None = None,
) -> list[tuple[str, int, Path]]:

    allowed_models   = {m.lower() for m in models_to_run}   if models_to_run   else None
    allowed_horizons = {int(h) for h in horizons_to_run}    if horizons_to_run else None

    items = []
    for p in optuna_dir.glob("hpo_*_H*.db"):
        m = _DB_RE.match(p.name)
        if not m:
            continue

        model, horizon = m.group("model"), int(m.group("horizon"))

        if allowed_models   and model.lower() not in allowed_models:
            continue
        if allowed_horizons and horizon not in allowed_horizons:
            continue

        items.append((model, horizon, p))

    return sorted(items, key=lambda x: (x[1], x[0].lower()))




def load_study_from_db(model: str, horizon: int, db_path: Path) -> optuna.Study:
    sname = f"HPO_{model}_H{horizon}"
    return optuna.load_study(study_name=sname, storage=f"sqlite:///{db_path}")


# =========================
# SPLITS: TRAIN+VAL combined, TEST untouched
# =========================
def make_final_splits_trainval_test(
    X, y, dates,
    window: int,
    horizon: int,
    batch_size: int,
    run_seed: int,
    seed_worker,
    num_workers: int = 0,
    pin_memory: bool = True,
):


    T = len(dates)
    train_end = int(T * 0.70)
    val_end   = int(T * 0.85)

    max_start = T - (window + horizon)
    train_start = 0
    train_stop  = min(train_end, max_start)

    val_start = train_stop
    val_stop  = min(val_end, max_start)

    test_start = val_stop
    test_stop  = T

    if train_stop <= train_start or val_stop <= val_start or test_stop <= test_start:
        raise ValueError(
            f"Invalid splits: train=({train_start},{train_stop}), "
            f"val=({val_start},{val_stop}), test=({test_start},{test_stop}), "
            f"T={T}, window={window}, horizon={horizon}"
        )

    # Final train uses [0, val_stop] to include both train and val
    final_train_start, final_train_stop = 0, val_stop
    splits = {
        "train_final": (final_train_start, final_train_stop),
        "test": (test_start, test_stop),
    }

    train_ds = SeqDataset(X, y, start=splits["train_final"][0], end=splits["train_final"][1],
                          window=window, horizon=horizon)
    test_ds  = SeqDataset(X, y, start=splits["test"][0], end=splits["test"][1],
                          window=window, horizon=horizon)

    g = torch.Generator().manual_seed(run_seed)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        worker_init_fn=seed_worker, generator=g,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        worker_init_fn=seed_worker, generator=g,
        num_workers=num_workers, pin_memory=pin_memory
    )
    return {"train_final": train_loader, "test": test_loader}


# =========================
# METRICS: F1-macro, minority precision/recall (classes 1,2)
# =========================
def minority_metrics(y_true: np.ndarray, y_pred: np.ndarray, minority_ids=(1, 2)) -> Dict[str, float]:
    out = {}
    # per-minority class precision/recall, plus average across minority classes
    precs, recs = [], []
    for cid, cname in zip(minority_ids, ["heatwave", "coldwave"]):
        yt = (y_true == cid).astype(int)
        yp = (y_pred == cid).astype(int)
        p = precision_score(yt, yp, zero_division=0)
        r = recall_score(yt, yp, zero_division=0)
        out[f"precision_{cname}"] = float(p)
        out[f"recall_{cname}"] = float(r)
        precs.append(p); recs.append(r)

    out["minority_precision_mean"] = float(np.mean(precs)) if precs else 0.0
    out["minority_recall_mean"] = float(np.mean(recs)) if recs else 0.0
    return out


# =========================
# APPLY BEST PARAMS -> CFG (robust mapping)
# =========================
def apply_best_params_to_cfg(CFG, best_params: Dict, horizon: int):
    """
    Matches the parameter names you used in Optuna objective:
      hidden_dim, dropout, window, batch_size, lr, weight_decay, w_heat, w_cold, num_layers
    """
    CFG.horizon = int(horizon)
    if "hidden_dim" in best_params:   CFG.hidden_dim = int(best_params["hidden_dim"])
    if "dropout" in best_params:      CFG.dropout = float(best_params["dropout"])
    if "window" in best_params:       CFG.window = int(best_params["window"])
    if "batch_size" in best_params:   CFG.batch_size = int(best_params["batch_size"])
    if "lr" in best_params:           CFG.lr = float(best_params["lr"])
    if "weight_decay" in best_params: CFG.weight_decay = float(best_params["weight_decay"])
    if "w_heat" in best_params:       CFG.w_heat = float(best_params["w_heat"])
    if "w_cold" in best_params:       CFG.w_cold = float(best_params["w_cold"])
    if "num_layers" in best_params:   CFG.num_layers = int(best_params["num_layers"])
    else:
        # schema consistency for GCN_FC or older studies
        CFG.num_layers = int(getattr(CFG, "num_layers", 1))
    return CFG


# =========================
# FINAL TRAIN
# =========================
def train_fixed_epochs(
    model: nn.Module,
    train_loader,
    CFG,
    edge_index,
    edge_weight=None,
    device="cpu",
    epochs: int = 20,
):
    model.to(device)
    model.train()

    # same weighting convention you used in train_model
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, float(CFG.w_heat), float(CFG.w_cold)], dtype=torch.float32).to(device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(CFG.lr), weight_decay=float(CFG.weight_decay))

    for ep in range(1, epochs + 1):
        tr_loss, tr_metrics = run_epoch(
            loader=train_loader,
            model=model,
            loss_fn=loss_fn,
            edge_index=edge_index,
            edge_weight=edge_weight,
            optimizer=optimizer,
            device=device
        )
        # keep prints minimal for HPC runs
        if ep == 1 or ep == epochs or (ep % 10 == 0):
            print(f"  epoch {ep:03d}/{epochs} | train loss={tr_loss:.4f} f1={tr_metrics['f1']:.3f}")

    return model


# =========================
# MAIN: load studies -> retrain -> evaluate -> save
# =========================
def run_final_retraining_from_optuna_dbs(
    optuna_dir: Path,
    CFG,
    device,
    edge_index,
    edge_weight,
    final_cfg: FinalRetrainConfig,
):
    set_global_seed(SEED)

    discovered = iter_hpo_dbs(optuna_dir)
    if not discovered:
        raise FileNotFoundError(f"No Optuna DBs found in: {optuna_dir}")

    rows = []

    for model_name, horizon, db_path in discovered:
        print(f"\n==============================")
        print(f"FINAL RETRAIN: {model_name} | H={horizon}")
        print(f"DB: {db_path}")
        print(f"==============================")

        # ---- Load study
        study = load_study_from_db(model_name, horizon, db_path)

        # ---- Extract best params & best epoch
        best_params = dict(study.best_params)
        bt = study.best_trial

        best_epoch = bt.user_attrs.get("best_epoch", None)
        if best_epoch is None:
            epochs = final_cfg.default_epochs
        else:
            be = int(best_epoch)
            epochs = (be + 1) if final_cfg.best_epoch_is_zero_based else be
            epochs = max(1, epochs)

        # ---- Apply best params to CFG
        from copy import deepcopy
        CFG_run = deepcopy(CFG)
        CFG_run = apply_best_params_to_cfg(CFG_run, best_params, horizon=horizon)

        # ---- Build loaders: Train+Val combined
        loaders = make_final_splits_trainval_test(
            X=X, y=y, dates=dates,
            window=int(CFG_run.window),
            horizon=int(CFG_run.horizon),
            batch_size=int(CFG_run.batch_size),
            run_seed=SEED,
            seed_worker=seed_worker,
            num_workers=final_cfg.num_workers,
            pin_memory=final_cfg.pin_memory
        )

        # ---- Build model with best config
        model = build_model(
            model_name=model_name,
            in_dim=int(CFG_run.input_dim),
            horizon=int(CFG_run.horizon),
            h=int(CFG_run.hidden_dim),
            dropout=float(CFG_run.dropout),
            n_classes=int(CFG_run.n_classes),
            num_layers=int(getattr(CFG_run, "num_layers", 1)),
            CFG=CFG_run
        ).to(device)

        # ---- Train fixed epochs
        model = train_fixed_epochs(
            model=model,
            train_loader=loaders["train_final"],
            CFG=CFG_run,
            edge_index=edge_index,
            edge_weight=edge_weight,
            device=device,
            epochs=epochs
        )

        # ---- Evaluate once on TEST
        met = evaluate(
            loader=loaders["test"],
            model=model,
            edge_index=edge_index,
            edge_weight=edge_weight,
            device=device
        )

        # ---- Extra minority metrics
        y_true = met["y_true"]
        y_pred = met["y_pred"]
        mm = minority_metrics(y_true, y_pred, minority_ids=(1, 2))

        # ---- Save weights
        weights_path = WEIGHTS_DIR / f"{model_name}_H{horizon}_best.pth"
        torch.save(
            {
                "model_name": model_name,
                "horizon": int(horizon),
                "best_params": best_params,
                "best_epoch": int(best_epoch) if best_epoch is not None else None,
                "epochs_trained_final": int(epochs),
                "state_dict": model.state_dict(),
                "cfg": {k: getattr(CFG_run, k) for k in dir(CFG_run) if not k.startswith("__") and not callable(getattr(CFG_run, k))},
            },
            weights_path
        )
        print(f"[OK] Saved weights: {weights_path}")

        # ---- Record row
        row = {
            "Model": model_name,
            "Horizon": int(horizon),
            "epochs_final": int(epochs),
            "best_val_f1_macro": float(study.best_value),
            "test_accuracy": float(met["accuracy"]),
            "test_f1_macro": float(met["f1_macro"]),
            "test_precision_macro": float(met["precision_macro"]),
            "test_recall_macro": float(met["recall_macro"]),
            **mm,
            "weights_path": str(weights_path),
            # store key training knobs used
            "window": int(CFG_run.window),
            "batch_size": int(CFG_run.batch_size),
            "lr": float(CFG_run.lr),
            "weight_decay": float(CFG_run.weight_decay),
            "hidden_dim": int(CFG_run.hidden_dim),
            "dropout": float(CFG_run.dropout),
            "w_heat": float(CFG_run.w_heat),
            "w_cold": float(CFG_run.w_cold),
            "num_layers": int(getattr(CFG_run, "num_layers", 1)),
        }
        rows.append(row)

        # ---- cleanup
        del model, loaders
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows).sort_values(["Horizon", "test_f1_macro"], ascending=[True, False]).reset_index(drop=True)
    df.to_csv(FINAL_CSV, index=False, float_format="%.6f")
    print(f"\n[OK] Saved final benchmark CSV: {FINAL_CSV}")

    # optional: also save JSON for manuscript appendix
    with open(OUT_DIR / "final_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)
    print(f"[OK] Saved JSON: {OUT_DIR / 'final_benchmark_results.json'}")

    return df


# -------------------------
# Helper: aggregate mean/std
# -------------------------
def summarize_runs(df_runs: pd.DataFrame, group_cols=("Model", "Horizon")) -> pd.DataFrame:
    """
    Aggregates numeric metrics columns into mean/std per group_cols.
    Keeps non-numeric columns via first() where reasonable (e.g., window, batch_size, etc.).
    """
    if df_runs.empty:
        return df_runs

    # numeric metrics to aggregate
    numeric_cols = df_runs.select_dtypes(include=[np.number]).columns.tolist()
    # do not aggregate identifiers
    numeric_cols = [c for c in numeric_cols if c not in list(group_cols) + ["run_id", "seed"]]

    agg = {c: ["mean", "std"] for c in numeric_cols}

    # keep "settings" columns (same across runs) as first
    keep_first = [
        "best_val_f1_macro", "epochs_final",
        "window", "batch_size", "lr", "weight_decay",
        "hidden_dim", "dropout", "w_heat", "w_cold", "num_layers"
    ]
    keep_first = [c for c in keep_first if c in df_runs.columns]
    for c in keep_first:
        agg[c] = ["first"]

    g = df_runs.groupby(list(group_cols), as_index=False).agg(agg)

    # flatten MultiIndex columns
    g.columns = [
        f"{a}_{b}" if b not in ("", None) else str(a)
        for a, b in g.columns.to_flat_index()
    ]

    # rename back group cols
    # (group cols will appear as "Model_" "Horizon_" due to flattening)
    rename_map = {}
    for c in group_cols:
        if f"{c}_" in g.columns:
            rename_map[f"{c}_"] = c
    g = g.rename(columns=rename_map)

    # optional: sort
    sort_cols = [c for c in ["Horizon", "test_f1_macro_mean"] if c in g.columns]
    if sort_cols:
        g = g.sort_values(sort_cols, ascending=[True, False] if len(sort_cols) == 2 else True).reset_index(drop=True)

    return g


# -------------------------
# MINIMAL change: extend existing function
# -------------------------

_DB_RE = re.compile(r"^hpo_(?P<model>.+)_H(?P<horizon>\d+)\.db$", re.IGNORECASE)

def iter_hpo_dbs(
    optuna_dir: Path,
    models_to_run: list[str] | None = None,
    horizons_to_run: list[int] | None = None,
) -> list[tuple[str, int, Path]]:

    allowed_models   = {m.lower() for m in models_to_run} if models_to_run else None
    allowed_horizons = {int(h) for h in horizons_to_run}  if horizons_to_run else None

    items = []
    for p in optuna_dir.glob("hpo_*_H*.db"):
        m = _DB_RE.match(p.name)
        if not m:
            continue

        model, horizon = m.group("model"), int(m.group("horizon"))

        if allowed_models and model.lower() not in allowed_models:
            continue
        if allowed_horizons and horizon not in allowed_horizons:
            continue

        items.append((model, horizon, p))

    return sorted(items, key=lambda x: (x[1], x[0].lower()))


N_RUNS = 10
RUN_SEEDS = [SEED + i for i in range(1, N_RUNS + 1)]



H_TAG = "H" + "-".join(map(str, HORIZONS))
FINAL_RUNS_CSV = OUT_DIR / f"{H_TAG}_final_benchmark_results_runs_event_level.csv"
FINAL_SUMMARY_CSV = OUT_DIR / f"{H_TAG}_final_benchmark_results_summary_event_level.csv"



def run_final_retraining_from_optuna_dbs_multi_run(
    optuna_dir: Path,
    CFG,
    device,
    edge_index,
    edge_weight,
    final_cfg: FinalRetrainConfig,
    run_seeds=RUN_SEEDS,
    models_to_run: list[str] | None = None,
):

    def _ts() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _safe_to_csv(df: pd.DataFrame, path: Path, **kwargs) -> Path:
        try:
            df.to_csv(path, **kwargs)
            return path
        except PermissionError:
            alt = path.with_name(f"{path.stem}_{_ts()}{path.suffix}")
            df.to_csv(alt, **kwargs)
            print(f"[WARN] Could not write (locked?): {path}")
            print(f"[OK]  Wrote instead: {alt}")
            return alt

    def _safe_json_dump(obj, path: Path) -> Path:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2)
            return path
        except PermissionError:
            alt = path.with_name(f"{path.stem}_{_ts()}{path.suffix}")
            with open(alt, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2)
            print(f"[WARN] Could not write (locked?): {path}")
            print(f"[OK]  Wrote instead: {alt}")
            return alt

    discovered = iter_hpo_dbs(optuna_dir, models_to_run=models_to_run, horizons_to_run=HORIZONS)
    if not discovered:
        raise FileNotFoundError(f"No Optuna DBs found in: {optuna_dir} (models_to_run={models_to_run})")

    def _best_window_from_db(model_name: str, horizon: int, db_path: Path) -> int:
        st = load_study_from_db(model_name, horizon, db_path)
        bp = dict(st.best_params)
        return int(bp.get("window", getattr(CFG, "window", 1)))

    global_max_window = max(_best_window_from_db(m, h, p) for (m, h, p) in discovered)
    global_max_window = max(1, int(global_max_window))

    # Reference test bounds (calendar indices), computed once
    ref_model, ref_h, ref_db = discovered[0]
    ref_study = load_study_from_db(ref_model, ref_h, ref_db)
    ref_best_params = dict(ref_study.best_params)

    CFG_ref = apply_best_params_to_cfg(deepcopy(CFG), ref_best_params, horizon=int(ref_h))
    ref_loaders = make_final_splits_trainval_test(
        X=X, y=y, dates=dates,
        window=int(CFG_ref.window),
        horizon=int(CFG_ref.horizon),
        batch_size=int(CFG_ref.batch_size),
        run_seed=int(run_seeds[0]),
        seed_worker=seed_worker,
        num_workers=int(final_cfg.num_workers),
        pin_memory=bool(final_cfg.pin_memory),
    )
    ref_test_ds = ref_loaders["test"].dataset
    if not hasattr(ref_test_ds, "start") or not hasattr(ref_test_ds, "end"):
        raise AttributeError("SeqDataset must define .start and .end for fixed calendar evaluation.")

    ref_start, ref_end = int(ref_test_ds.start), int(ref_test_ds.end)
    fixed_eval_start_day = ref_start + global_max_window
    fixed_eval_end_day = ref_end - 1
    del ref_loaders

    if fixed_eval_end_day < fixed_eval_start_day:
        raise ValueError(
            f"Invalid global eval range: start={fixed_eval_start_day}, end={fixed_eval_end_day}. "
            f"Increase test size or reduce global_max_window."
        )

    def _epochs_from_best_epoch(best_epoch) -> int:
        if best_epoch is None:
            return int(final_cfg.default_epochs)
        be = int(best_epoch)
        ep = (be + 1) if bool(final_cfg.best_epoch_is_zero_based) else be
        return max(1, int(ep))

    rows: list[dict] = []

    for model_name, horizon, db_path in discovered:
        print(f"\n==============================")
        print(f"FINAL RETRAIN (multi-run): {model_name} | H={horizon}")
        print(f"DB: {db_path}")
        print(f"==============================")

        study = load_study_from_db(model_name, horizon, db_path)
        best_params = dict(study.best_params)
        bt = study.best_trial
        epochs = _epochs_from_best_epoch(bt.user_attrs.get("best_epoch", None))

        CFG_base = apply_best_params_to_cfg(deepcopy(CFG), best_params, horizon=int(horizon))

        for run_id, seed in enumerate(run_seeds, start=1):
            print(f"\n--- Run {run_id}/{len(run_seeds)} | seed={seed} ---")
            set_global_seed(int(seed))

            loaders = make_final_splits_trainval_test(
                X=X, y=y, dates=dates,
                window=int(CFG_base.window),
                horizon=int(CFG_base.horizon),
                batch_size=int(CFG_base.batch_size),
                run_seed=int(seed),
                seed_worker=seed_worker,
                num_workers=int(final_cfg.num_workers),
                pin_memory=bool(final_cfg.pin_memory),
            )

            test_ds = loaders["test"].dataset
            if int(test_ds.start) != ref_start or int(test_ds.end) != ref_end:
                raise ValueError(
                    f"Test split mismatch across configs! "
                    f"ref=({ref_start},{ref_end}), this=({int(test_ds.start)},{int(test_ds.end)})."
                )

            model = build_model(
                model_name=model_name,
                in_dim=int(CFG_base.input_dim),
                horizon=int(CFG_base.horizon),
                h=int(CFG_base.hidden_dim),
                dropout=float(CFG_base.dropout),
                n_classes=int(CFG_base.n_classes),
                num_layers=int(getattr(CFG_base, "num_layers", 1)),
                CFG=CFG_base,
            ).to(device)

            model = train_fixed_epochs(
                model=model,
                train_loader=loaders["train_final"],
                CFG=CFG_base,
                edge_index=edge_index,
                edge_weight=edge_weight,
                device=device,
                epochs=int(epochs),
            )

            met = evaluate(
                loader=loaders["test"],
                model=model,
                edge_index=edge_index,
                edge_weight=edge_weight,
                device=device,
                minlen=3,
                min_overlap=1,
                eval_start_day=int(fixed_eval_start_day),
                eval_end_day=int(fixed_eval_end_day),
                y_global=y,
            )

            mm = minority_metrics(met["y_true"], met["y_pred"], minority_ids=(1, 2))
            evt = met.get("event_level", {}) or {}
            hw = evt.get("HW", {}) or {}
            cw = evt.get("CW", {}) or {}

            weights_path = WEIGHTS_DIR / f"{model_name}_H{int(horizon)}_run{run_id:02d}.pth"
            torch.save(
                {
                    "model_name": model_name,
                    "horizon": int(horizon),
                    "run_id": int(run_id),
                    "seed": int(seed),
                    "best_params": best_params,
                    "best_epoch": bt.user_attrs.get("best_epoch", None),
                    "epochs_trained_final": int(epochs),
                    "state_dict": model.state_dict(),
                    "eval_start_day": int(fixed_eval_start_day),
                    "eval_end_day": int(fixed_eval_end_day),
                    "global_max_window": int(global_max_window),
                },
                weights_path,
            )
            print(f"[OK] Saved weights: {weights_path.name}")

            rows.append({
                "Model": model_name,
                "Horizon": int(horizon),
                "run_id": int(run_id),
                "seed": int(seed),

                "epochs_final": int(epochs),
                "best_val_f1_macro": float(study.best_value),

                "eval_start_day": int(fixed_eval_start_day),
                "eval_end_day": int(fixed_eval_end_day),
                "global_max_window": int(global_max_window),

                "test_accuracy": float(met["accuracy"]),
                "test_f1_macro": float(met["f1_macro"]),
                "test_precision_macro": float(met["precision_macro"]),
                "test_recall_macro": float(met["recall_macro"]),
                **mm,

                "event_precision_hw": float(hw.get("event_precision", 0.0)),
                "event_recall_hw": float(hw.get("event_recall", 0.0)),
                "event_f1_hw": float(hw.get("event_f1", 0.0)),
                "n_true_events_hw": int(hw.get("n_true_events", 0)),
                "n_pred_events_hw": int(hw.get("n_pred_events", 0)),
                "n_detected_events_hw": int(hw.get("n_detected_events", 0)),

                "event_precision_cw": float(cw.get("event_precision", 0.0)),
                "event_recall_cw": float(cw.get("event_recall", 0.0)),
                "event_f1_cw": float(cw.get("event_f1", 0.0)),
                "n_true_events_cw": int(cw.get("n_true_events", 0)),
                "n_pred_events_cw": int(cw.get("n_pred_events", 0)),
                "n_detected_events_cw": int(cw.get("n_detected_events", 0)),

                "window": int(CFG_base.window),
                "batch_size": int(CFG_base.batch_size),
                "lr": float(CFG_base.lr),
                "weight_decay": float(CFG_base.weight_decay),
                "hidden_dim": int(CFG_base.hidden_dim),
                "dropout": float(CFG_base.dropout),
                "w_heat": float(CFG_base.w_heat),
                "w_cold": float(CFG_base.w_cold),
                "num_layers": int(getattr(CFG_base, "num_layers", 1)),

                "weights_path": str(weights_path),
            })

            del model, loaders
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    df_runs = pd.DataFrame(rows)
    _safe_to_csv(df_runs, FINAL_RUNS_CSV, index=False, float_format="%.6f")
    print(f"\n[OK] Saved per-run CSV: {FINAL_RUNS_CSV}")

    df_summary = summarize_runs(df_runs, group_cols=("Model", "Horizon"))
    _safe_to_csv(df_summary, FINAL_SUMMARY_CSV, index=False, float_format="%.6f")
    print(f"[OK] Saved summary CSV (mean/std): {FINAL_SUMMARY_CSV}")

    _safe_json_dump(df_runs.to_dict(orient="records"), OUT_DIR / "final_benchmark_results_runs_event_level.json")

    return df_runs, df_summary


# =========================
# RUN (multi-run final retraining)
# =========================
models_subset = ["ASTGCN", "DCRNN", "TransformerModel", "GCN_FC", "STGCN_GRU", "FC_LSTM", "TCN", "CONVGRU", "CONVLSTM"]

df_runs, df_summary = run_final_retraining_from_optuna_dbs_multi_run(
    optuna_dir=OPTUNA_DIR,
    CFG=CFG,
    device=DEVICE,
    edge_index=edge_index,
    edge_weight=edge_weight,
    final_cfg=FinalRetrainConfig(),
    run_seeds=RUN_SEEDS,
    models_to_run=models_subset
)