#!/usr/bin/env python3
"""Train the cloudnr neural L_rest MLP (docs/architecture.md section 8).

Function approximated:
    L_rest(x, d) = f_sun * E_sun + f_sky * L_sky + f_gnd * L_gnd
the in-scattered radiance at medium point x toward -d (d = view/tracing direction),
EXCLUDING the direct single-scattered sun term, which the renderer's integrator adds
itself. f = (f_sun, f_sky, f_gnd) are scalars (grey medium) for unit sources.

Inputs are the 40-float descriptors written by `cloudnr dump-training` (.cnrd). The MLP
40->32->32->32->3 (ReLU) predicts y with f = expm1(max(y, 0)); the loss is MSE between y
and log1p(f_target).

Usage:
    .venv/bin/python tools/train.py --train data/train.cnrd --val data/val.cnrd --out models/<name>

Writes models/<name>.cnrm (fp16 weights), models/<name>.json (sidecar metadata) and
models/<name>.check.bin (10000 validation inputs with the outputs of the exported fp16
weights evaluated in fp32, for `cloudnr neural-check`).
"""

import argparse
import hashlib
import json
import math
import platform
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

CNRD_FORMAT = "<4sIIIQIIIffIII8x"  # 64-byte header, src/neural_types.h CnrdHeader
CNRM_FORMAT = "<4sIIIIIffI"  # 36-byte header, src/neural_types.h CnrmHeader
CNRC_FORMAT = "<4sIII"  # check file: magic, count, n_in, n_out
N_IN, N_HIDDEN, N_LAYERS, N_OUT, LEVELS = 40, 32, 3, 3, 5
FORMAT_VERSION = 1
FIRST_TRAINING_SEED = 1000  # density seeds 1-999 are reserved for evaluation
CHECK_COUNT = 10000

assert struct.calcsize(CNRD_FORMAT) == 64
assert struct.calcsize(CNRM_FORMAT) == 36


def load_cnrd(path):
    data = Path(path).read_bytes()
    if len(data) < 64:
        raise ValueError(f"{path}: too small for a CNRD header")
    fields = struct.unpack_from(CNRD_FORMAT, data, 0)
    keys = ["magic", "version", "n_in", "n_out", "n_records", "seed_begin", "seed_end", "paths", "hg_g",
            "albedo", "shadow_steps", "n_scenes", "records_per_scene"]
    header = dict(zip(keys, fields))
    if header["magic"] != b"CNRD":
        raise ValueError(f"{path}: bad magic {header['magic']!r}")
    if header["version"] != 1:
        raise ValueError(f"{path}: unsupported version {header['version']}")
    if header["n_in"] != N_IN or header["n_out"] != N_OUT:
        raise ValueError(f"{path}: dimensions {header['n_in']}x{header['n_out']} != {N_IN}x{N_OUT}")
    stride = N_IN + N_OUT
    expected = 64 + header["n_records"] * stride * 4
    if len(data) != expected:
        raise ValueError(f"{path}: size {len(data)} != expected {expected}")
    if header["n_records"] == 0:
        raise ValueError(f"{path}: no records")
    records = np.frombuffer(data, dtype="<f4", offset=64).reshape(header["n_records"], stride)
    if not np.isfinite(records).all():
        raise ValueError(f"{path}: non-finite values")
    targets = records[:, N_IN:]
    if (targets < 0).any():
        raise ValueError(f"{path}: negative target coefficients")
    del header["magic"]
    header["path"] = str(path)
    header["sha256"] = hashlib.sha256(data).hexdigest()
    return records[:, :N_IN].astype(np.float32), targets.astype(np.float32), header


def check_datasets(train, val):
    for name, h in (("train", train), ("val", val)):
        if h["seed_begin"] < FIRST_TRAINING_SEED:
            raise ValueError(f"{name} set uses evaluation density seeds (seed_begin {h['seed_begin']} < 1000)")
    if train["seed_begin"] < val["seed_end"] and val["seed_begin"] < train["seed_end"]:
        raise ValueError("train and val density seed ranges overlap")
    if abs(train["hg_g"] - val["hg_g"]) > 1e-6 or abs(train["albedo"] - val["albedo"]) > 1e-6:
        raise ValueError("train and val sets differ in hg_g/albedo")
    if train["shadow_steps"] != val["shadow_steps"]:
        raise ValueError("train and val sets differ in shadow_steps")


def make_model():
    layers = []
    width = N_IN
    for _ in range(N_LAYERS):
        layers += [nn.Linear(width, N_HIDDEN), nn.ReLU()]
        width = N_HIDDEN
    layers.append(nn.Linear(width, N_OUT))
    return nn.Sequential(*layers)


def linears(model):
    return [m for m in model if isinstance(m, nn.Linear)]


def fp16_weights(model):
    """Per layer: W[out][in] row-major, then bias[out]; fp16 little-endian."""
    parts = []
    for layer in linears(model):
        parts.append(layer.weight.detach().cpu().numpy().astype(np.float32).reshape(-1))
        parts.append(layer.bias.detach().cpu().numpy().astype(np.float32).reshape(-1))
    weights = np.concatenate(parts).astype("<f2")
    if not np.isfinite(weights.astype(np.float32)).all():
        raise ValueError("weights overflow fp16")
    return weights


def quantized_copy(model):
    """CPU fp32 model whose parameters are the exported fp16 values."""
    q = make_model()
    q.load_state_dict({k: v.detach().cpu().half().float() for k, v in model.state_dict().items()})
    return q.eval()


@torch.no_grad()
def predict_f(model, x):
    y = model(x)
    return torch.expm1(torch.clamp(y, min=0.0))


@torch.no_grad()
def mean_loss(model, x, y_log, batch=65536):
    total = torch.zeros((), device=x.device)
    for start in range(0, x.shape[0], batch):
        pred = model(x[start:start + batch])
        total += ((pred - y_log[start:start + batch]) ** 2).sum()
    return float(total) / (x.shape[0] * N_OUT)


def rel_errors(a, b):
    rel = np.abs(a - b) / (np.abs(b) + 1e-3)
    return float(rel.max()), float(rel.mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--out", required=True, help="output prefix, e.g. models/cloud_mlp")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1, help="torch seed (init and shuffling)")
    parser.add_argument("--cpu", action="store_true", help="force CPU even if MPS is available")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1:
        parser.error("--epochs and --batch must be >= 1")

    # Output prefix; ".cnrm", ".json" and ".check.bin" are appended (names may contain dots).
    out = str(args.out)
    if out.endswith(".cnrm"):
        out = out[:-len(".cnrm")]
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    x_train, f_train, train_header = load_cnrd(args.train)
    x_val, f_val, val_header = load_cnrd(args.val)
    check_datasets(train_header, val_header)
    print(f"train {len(x_train)} records (seeds {train_header['seed_begin']}-{train_header['seed_end'] - 1}), "
          f"val {len(x_val)} records (seeds {val_header['seed_begin']}-{val_header['seed_end'] - 1})", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() and not args.cpu else "cpu")
    xt = torch.from_numpy(x_train).to(device)
    yt = torch.log1p(torch.from_numpy(f_train)).to(device)
    xv = torch.from_numpy(x_val).to(device)
    yv = torch.log1p(torch.from_numpy(f_val)).to(device)

    model = make_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps_per_epoch = math.ceil(len(x_train) / args.batch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * steps_per_epoch)
    shuffle = torch.Generator().manual_seed(args.seed)

    train_curve, val_curve = [], []
    best_val, best_epoch, best_state = math.inf, -1, None
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(x_train), generator=shuffle).to(device)
        running = torch.zeros((), device=device)
        for step in range(steps_per_epoch):
            idx = perm[step * args.batch:(step + 1) * args.batch]
            pred = model(xt[idx])
            loss = nn.functional.mse_loss(pred, yt[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.detach() * idx.shape[0]
        model.eval()
        train_loss = float(running) / len(x_train)
        val_loss = mean_loss(model, xv, yv)
        train_curve.append(train_loss)
        val_curve.append(val_loss)
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch + 1}/{args.epochs} train {train_loss:.6g} val {val_loss:.6g} "
              f"lr {scheduler.get_last_lr()[0]:.3g} ({time.time() - start:.0f}s)", flush=True)

    best = make_model()
    best.load_state_dict(best_state)
    best.eval()
    weights = fp16_weights(best)
    quant = quantized_copy(best)

    hg_g, albedo = train_header["hg_g"], train_header["albedo"]
    cnrm_path = Path(out + ".cnrm")
    cnrm = struct.pack(CNRM_FORMAT, b"CNRM", FORMAT_VERSION, N_IN, N_HIDDEN, N_LAYERS, N_OUT, hg_g, albedo,
                       LEVELS) + weights.tobytes()
    cnrm_path.write_bytes(cnrm)

    # Parity data: outputs of the exported fp16 weights evaluated in fp32 on the CPU.
    count = min(CHECK_COUNT, len(x_val))
    x_check = torch.from_numpy(x_val[:count].copy())
    f_check = predict_f(quant, x_check).numpy().astype("<f4")
    f_fp32 = predict_f(best, x_check).numpy()
    check_path = Path(out + ".check.bin")
    check_path.write_bytes(struct.pack(CNRC_FORMAT, b"CNRC", count, N_IN, N_OUT) +
                           x_check.numpy().astype("<f4").tobytes() + f_check.tobytes())
    quant_max, quant_mean = rel_errors(f_check, f_fp32)
    val_loss_fp16 = mean_loss(quant, torch.from_numpy(x_val), torch.log1p(torch.from_numpy(f_val)))

    def dataset_info(h):
        return {
            "path": h["path"], "sha256": h["sha256"], "n_records": h["n_records"],
            "seed_range": [h["seed_begin"], h["seed_end"] - 1], "paths": h["paths"],
            "shadow_steps": h["shadow_steps"], "n_scenes": h["n_scenes"],
            "records_per_scene": h["records_per_scene"], "hg_g": h["hg_g"], "albedo": h["albedo"],
        }

    sidecar = {
        "format_version": FORMAT_VERSION,
        "model": cnrm_path.name,
        "cnrm_sha256": hashlib.sha256(cnrm).hexdigest(),
        "function": "L_rest(x,d) = f_sun*E_sun + f_sky*L_sky + f_gnd*L_gnd, in-scattered radiance at x toward -d "
                    "excluding direct single-scattered sun",
        "architecture": {
            "n_in": N_IN, "n_hidden": N_HIDDEN, "n_hidden_layers": N_LAYERS, "n_out": N_OUT, "levels": LEVELS,
            "activation": "relu", "output": "f = expm1(max(y, 0)), y trained against log1p(f)",
            "weights": "fp16, per layer W[out][in] row-major then bias[out]",
            "inference": "fp16 weights, fp32 activations and accumulation (MSL)",
        },
        "medium": {"hg_g": hg_g, "albedo": albedo},
        "hyperparameters": {
            "optimizer": "adam", "lr": args.lr, "schedule": "cosine per step", "batch": args.batch,
            "epochs": args.epochs, "loss": "mse(y, log1p(f))", "torch_seed": args.seed, "device": str(device),
        },
        "train": dataset_info(train_header),
        "val": dataset_info(val_header),
        "train_seed_range": [train_header["seed_begin"], train_header["seed_end"] - 1],
        "val_seed_range": [val_header["seed_begin"], val_header["seed_end"] - 1],
        "versions": {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version()},
        "renderer_version": "nogit",
        "loss_curves": {"train": train_curve, "val": val_curve},
        "best_epoch": best_epoch + 1,
        "best_val_loss": best_val,
        "val_loss_fp16_weights": val_loss_fp16,
        "fp16_quantization_rel_error": {"max": quant_max, "mean": quant_mean, "count": count},
        "check_file": check_path.name,
        "train_seconds": time.time() - start,
        "known_bias": f"targets are {train_header['paths']}-path MC estimates; the log-space loss is biased by "
                      "target noise (E[log1p(f_hat)] != log1p(f))",
    }
    json_path = Path(out + ".json")
    json_path.write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"wrote {cnrm_path} ({len(weights)} fp16 values), {json_path}, {check_path}; "
          f"best epoch {best_epoch + 1} val {best_val:.6g}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
