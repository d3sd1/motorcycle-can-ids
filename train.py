#!/usr/bin/env python3
"""
Train autoencoder and baseline models for CAN bus anomaly detection.

Paper: Lightweight Autoencoder-Based Anomaly Detection for CAN Bus
       in Competition Motorcycles Deployed on ARM Cortex-M7
"""

import json
import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
import pickle
from pathlib import Path
from typing import Tuple, Dict, List

CONFIG_PATH = Path(__file__).parent / "config.json"
with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

RESULTS_DIR = Path(__file__).parent / CONFIG["output"]["results_dir"]
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CANAutoencoder(nn.Module):
    """Lightweight autoencoder for CAN traffic anomaly detection.

    Architecture: input_dim -> d/2 -> d/4 -> latent_dim -> d/4 -> d/2 -> input_dim
    For d=80, l=10: 80 -> 40 -> 20 -> 10 -> 20 -> 40 -> 80
    """

    def __init__(self, input_dim: int = 80, latent_dim: int = 10):
        super().__init__()
        h1 = input_dim // 2   # 40
        h2 = input_dim // 4   # 20

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.ReLU(),
            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.Linear(h1, input_dim),
            # Linear output activation (no activation)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def encode(self, x):
        return self.encoder(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def get_layer_info(self):
        """Return info about each layer for resource estimation."""
        layers = []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                layers.append({
                    "name": name,
                    "type": "Linear",
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                    "params": module.in_features * module.out_features + module.out_features,
                    "macs": module.in_features * module.out_features,
                })
        return layers


class LSTMAutoencoder(nn.Module):
    """LSTM-based autoencoder for sequence-level CAN anomaly detection.

    This is a baseline method -- more powerful but too large for Cortex-M7.
    """

    def __init__(self, input_dim: int = 80, hidden_dim: int = 32,
                 seq_len: int = 5, dropout: float = 0.15):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True,
                                    dropout=0)
        self.dropout = nn.Dropout(dropout)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True,
                                    dropout=0)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (h, c) = self.encoder_lstm(x)
        h = self.dropout(h)
        decoder_input = h.permute(1, 0, 2).repeat(1, self.seq_len, 1)
        decoder_out, _ = self.decoder_lstm(decoder_input, (h, c))
        decoder_out = self.dropout(decoder_out)
        output = self.output_layer(decoder_out)
        return output

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


def train_autoencoder(X_train: np.ndarray, X_val: np.ndarray,
                      seed: int, config: dict, verbose: bool = True) -> nn.Module:
    """Train the autoencoder with a specific seed."""
    set_all_seeds(seed)

    ae_config = config["autoencoder"]
    input_dim = X_train.shape[1]
    latent_dim = ae_config["latent_dim"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CANAutoencoder(input_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ae_config["learning_rate"])
    criterion = nn.MSELoss()

    train_tensor = torch.FloatTensor(X_train).to(device)
    val_tensor = torch.FloatTensor(X_val).to(device)

    train_dataset = TensorDataset(train_tensor)
    train_loader = DataLoader(
        train_dataset, batch_size=ae_config["batch_size"], shuffle=True,
        drop_last=False,
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    train_losses = []
    val_losses = []

    for epoch in range(ae_config["max_epochs"]):
        # Training
        model.train()
        train_loss = 0.0
        n_batches = 0
        for (batch,) in train_loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(val_tensor)
            val_loss = criterion(val_output, val_tensor).item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= ae_config["early_stopping_patience"]:
                if verbose:
                    print(f"    Early stopping at epoch {epoch + 1}, "
                          f"best val loss: {best_val_loss:.6f}")
                break

        if verbose and (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}: train_loss={train_loss:.6f}, "
                  f"val_loss={val_loss:.6f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model = model.cpu()

    return model, {"train_losses": train_losses, "val_losses": val_losses,
                   "best_epoch": len(train_losses) - patience_counter,
                   "final_val_loss": best_val_loss}


def calibrate_threshold(model: nn.Module, X_val: np.ndarray,
                        percentile: int = 99) -> Tuple:
    """Calibrate anomaly threshold on validation set.

    Returns: (threshold, all_errors)
    """
    model.eval()
    with torch.no_grad():
        val_tensor = torch.FloatTensor(X_val)
        reconstructed = model(val_tensor)
        errors = torch.mean((val_tensor - reconstructed) ** 2, dim=1).numpy()

    threshold = float(np.percentile(errors, percentile))
    return threshold, errors


def quantize_model_int8(model: nn.Module, X_calibration: np.ndarray) -> dict:
    """Simulate INT8 quantization of the autoencoder.

    Instead of using torch.quantization (which has limited support for some
    architectures), we simulate INT8 quantization by:
    1. Computing per-layer min/max of weights and activations
    2. Quantizing weights to INT8
    3. Running inference with quantized weights
    4. Reporting accuracy impact

    Returns dict with quantized model info and performance.
    """
    model.eval()

    # Collect weight statistics
    weight_stats = {}
    total_params = 0
    for name, param in model.named_parameters():
        w = param.data.numpy()
        w_min, w_max = float(w.min()), float(w.max())
        # INT8 quantization: scale and zero-point
        scale = (w_max - w_min) / 255.0 if w_max != w_min else 1.0
        zero_point = int(round(-w_min / scale)) if scale != 0 else 0
        # Quantize
        w_int8 = np.clip(np.round(w / scale) + zero_point, 0, 255).astype(np.uint8)
        # Dequantize
        w_deq = (w_int8.astype(np.float32) - zero_point) * scale

        weight_stats[name] = {
            "shape": list(w.shape),
            "n_params": int(w.size),
            "min": w_min,
            "max": w_max,
            "scale": scale,
            "zero_point": zero_point,
            "quantization_error_mse": float(np.mean((w - w_deq) ** 2)),
        }
        total_params += w.size

    # Create quantized copy of model for inference comparison
    quantized_state = {}
    for name, param in model.named_parameters():
        w = param.data.numpy()
        stats = weight_stats[name]
        scale = stats["scale"]
        zp = stats["zero_point"]
        w_int8 = np.clip(np.round(w / scale) + zp, 0, 255).astype(np.uint8)
        w_deq = (w_int8.astype(np.float32) - zp) * scale
        quantized_state[name] = torch.FloatTensor(w_deq)

    # Run calibration inference with quantized weights
    quantized_model = CANAutoencoder(model.encoder[0].in_features,
                                     model.encoder[4].out_features)
    state_dict = quantized_model.state_dict()
    for name in quantized_state:
        if name in state_dict:
            state_dict[name] = quantized_state[name]
    quantized_model.load_state_dict(state_dict)
    quantized_model.eval()

    with torch.no_grad():
        cal_tensor = torch.FloatTensor(X_calibration)
        # Original model output
        orig_output = model(cal_tensor)
        # Quantized model output
        quant_output = quantized_model(cal_tensor)

    # Compute quantization degradation
    orig_errors = torch.mean((cal_tensor - orig_output) ** 2, dim=1).numpy()
    quant_errors = torch.mean((cal_tensor - quant_output) ** 2, dim=1).numpy()

    return {
        "total_params": total_params,
        "fp32_model_size_bytes": total_params * 4,
        "int8_model_size_bytes": total_params * 1,
        "fp32_model_size_kb": round(total_params * 4 / 1024, 2),
        "int8_model_size_kb": round(total_params * 1 / 1024, 2),
        "weight_stats": weight_stats,
        "quantized_model": quantized_model,
        "orig_recon_error_mean": float(np.mean(orig_errors)),
        "quant_recon_error_mean": float(np.mean(quant_errors)),
        "size_reduction_ratio": 4.0,
    }


def train_ocsvm(X_train: np.ndarray, seed: int, config: dict) -> object:
    """Train One-Class SVM baseline."""
    ocsvm_config = config["baselines"]["ocsvm"]
    # Subsample for OC-SVM if dataset is too large (performance)
    max_samples = 10000
    if len(X_train) > max_samples:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(X_train), max_samples, replace=False)
        X_sub = X_train[indices]
    else:
        X_sub = X_train

    model = OneClassSVM(
        kernel=ocsvm_config["kernel"],
        nu=ocsvm_config["nu"],
        gamma=ocsvm_config["gamma"],
    )
    model.fit(X_sub)
    return model


def train_isolation_forest(X_train: np.ndarray, seed: int, config: dict) -> object:
    """Train Isolation Forest baseline."""
    if_config = config["baselines"]["isolation_forest"]
    contam = if_config["contamination"]
    if isinstance(contam, str):
        contam = contam  # "auto" is valid for sklearn
    model = IsolationForest(
        n_estimators=if_config["n_estimators"],
        contamination=contam,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model


def train_lstm_ae(X_train_seq: np.ndarray, X_val_seq: np.ndarray,
                  seed: int, config: dict, verbose: bool = True):
    """Train LSTM autoencoder baseline."""
    set_all_seeds(seed)
    lstm_config = config["baselines"]["lstm_ae"]
    input_dim = X_train_seq.shape[2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=lstm_config["hidden_units"],
        seq_len=lstm_config["sequence_length"],
        dropout=lstm_config.get("dropout", 0.15),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lstm_config["learning_rate"])
    criterion = nn.MSELoss()

    train_tensor = torch.FloatTensor(X_train_seq).to(device)
    val_tensor = torch.FloatTensor(X_val_seq).to(device)

    train_dataset = TensorDataset(train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(lstm_config["max_epochs"]):
        model.train()
        for (batch,) in train_loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_output = model(val_tensor)
            val_loss = criterion(val_output, val_tensor).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= lstm_config["early_stopping_patience"]:
                if verbose:
                    print(f"    LSTM-AE early stopping at epoch {epoch + 1}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model = model.cpu()

    return model


def create_sequences(X: np.ndarray, seq_len: int = 10) -> np.ndarray:
    """Create sequences of windows for LSTM-AE.

    Returns: (n_sequences, seq_len, n_features)
    """
    if len(X) < seq_len:
        return np.zeros((0, seq_len, X.shape[1]))

    sequences = []
    for i in range(len(X) - seq_len + 1):
        sequences.append(X[i:i + seq_len])
    return np.array(sequences)


if __name__ == "__main__":
    print("=" * 60)
    print("CAN Bus Anomaly Detection -- Model Info")
    print("=" * 60)

    model = CANAutoencoder(80, 10)
    n_params = model.count_parameters()
    print(f"\nAutoencoder architecture: 80 -> 40 -> 20 -> 10 -> 20 -> 40 -> 80")
    print(f"Total parameters: {n_params}")
    print(f"FP32 model size: {n_params * 4 / 1024:.2f} KB")
    print(f"INT8 model size:  {n_params * 1 / 1024:.2f} KB")

    print(f"\nLayer details:")
    for info in model.get_layer_info():
        print(f"  {info['name']}: {info['in_features']}x{info['out_features']} "
              f"= {info['params']} params, {info['macs']} MACs")

    lstm = LSTMAutoencoder(80, 64, 10)
    print(f"\nLSTM-AE parameters: {lstm.count_parameters()}")
