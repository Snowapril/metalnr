#!/usr/bin/env python3
"""Train the cloudnr neural shader (the network of "Neural Shaders, Explained With Clouds").

Function approximated:
    Ls(x, d) = f_sun * E_sun + f_sky * L_sky + f_gnd * L_gnd
the FULL in-scattered radiance at medium point x toward -d (d = view/tracing direction), before
albedo, including the single-scattered sun. The medium is grey, so f = (f_sun, f_sky, f_gnd) are
scalars: the in-scattered radiance for unit sun irradiance / sky radiance / ground radiance. In the
renderer's ray march this is the one line the network replaces; the march applies albedo and
extinction. Targets come from `cloudnr dump-training` (.cnrd v2): Monte Carlo averages over N paths
of [HG(sun, d) * T_sun(x) + random walk from x along w ~ HG(d)], at the first collision of random
camera rays with random cumulus shapes (random sun, sigma_t_max log-uniform).

Inputs: the 13-float descriptor of src/shaders/neural_common.metal build_descriptor: sigma_t at x
for radii 6..192 m (mip lod log2(r / voxel)), sigma_t at x + r * sun_dir for the same radii, and
cos(view, sun).

Network (fixed): standardize x = (desc - mean) * scale (training-set statistics, scale = 1/std with
std floored at 1e-6); 13 -> 16 -> 16 -> 3 with squared ReLU max(z, 0)^2 on the hidden layers and a
squared output y = z^2 (non-negative radiance); at inference f = min(y, clamp) with clamp = 1.5 x
the largest training target per output. Loss: MSE on the RAW targets (a log / cube-root transform
of the targets biases the result, as the article shows). Adam with cosine learning-rate decay; the
checkpoint with the best validation loss is exported.

Usage:
    .venv/bin/python tools/train.py --train data/train.cnrd --val data/val.cnrd --out models/cumulus_net

Writes models/<name>.cnrm (format_version 2, fp32 weights), models/<name>.json (sidecar metadata,
loss curves, held-out metrics incl. per-output Pearson correlation) and models/<name>.check.bin
(CNRC v2: 10000 validation inputs and the fp32 PyTorch outputs, for `cloudnr neural-check`).
"""

import argparse
import hashlib
import json
import math
import platform
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

CNRD_FORMAT = "<4sIIIQIIIIIffffI"  # 64 bytes, src/neural_types.h CnrdHeader (version 2)
CNRM_FORMAT = "<4sIIIIIff6f13f13f3f"  # 172 bytes, src/neural_types.h CnrmHeader (format_version 2)
CNRC_FORMAT = "<4sIIII"  # 20 bytes, src/neural_types.h CnrcHeader (version 2)
N_IN, N_HIDDEN, N_HIDDEN_LAYERS, N_OUT = 13, 16, 2, 3
RADII = [6.0, 12.0, 24.0, 48.0, 96.0, 192.0]
CNRD_VERSION = 2
FORMAT_VERSION = 2
CHECK_VERSION = 2
FIRST_TRAINING_SEED = 1000  # shape seeds 1-999 are reserved for evaluation
CHECK_COUNT = 10000
STD_FLOOR = 1e-6
CLAMP_FACTOR = 1.5
OUTPUT_NAMES = ["sun", "sky", "ground"]
INPUT_NAMES = [f"local_{int(r)}m" for r in RADII] + [f"sun_{int(r)}m" for r in RADII] + ["cos_view_sun"]

assert struct.calcsize(CNRD_FORMAT) == 64
assert struct.calcsize(CNRM_FORMAT) == 172
assert struct.calcsize(CNRC_FORMAT) == 20


def load_cnrd(path):
    data = Path(path).read_bytes()
    if len(data) < 64:
        raise ValueError(f"{path}: too small for a CNRD header")
    fields = struct.unpack_from(CNRD_FORMAT, data, 0)
    keys = ["magic", "version", "n_in", "n_out", "n_records", "shape_begin", "shape_end", "n_shapes",
            "records_per_shape", "paths", "hg_g", "albedo", "sigma_t_min", "sigma_t_max", "fixed_shape"]
    header = dict(zip(keys, fields))
    if header["magic"] != b"CNRD":
        raise ValueError(f"{path}: bad magic {header['magic']!r}")
    if header["version"] != CNRD_VERSION:
        raise ValueError(f"{path}: unsupported version {header['version']} (expected {CNRD_VERSION})")
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
    span = header["shape_end"] - header["shape_begin"]
    # Fixed-shape datasets reuse one template cloud; shape seeds then only salt the sun sampling.
    header["shape_ids"] = ([] if header["fixed_shape"] else
                           sorted({header["shape_begin"] + k % span for k in range(header["n_shapes"])}))
    return records[:, :N_IN].astype(np.float32), targets.astype(np.float32), header


def check_datasets(train, val):
    if train["fixed_shape"] != val["fixed_shape"]:
        raise ValueError("train and val sets must both be fixed-shape or both random-shape")
    if train["fixed_shape"]:
        # Single-cloud weights (the article's sun-sweep setup): the template cloud is the evaluation cloud by
        # design and the sidecar records fixed_shape; only the sun sampling seeds must differ.
        if train["shape_begin"] == val["shape_begin"]:
            raise ValueError("fixed-shape train and val sets use the same sun sampling seed")
    else:
        for name, h in (("train", train), ("val", val)):
            if h["shape_begin"] < FIRST_TRAINING_SEED:
                raise ValueError(f"{name} set uses evaluation shape seeds (shape_begin {h['shape_begin']} < 1000)")
        if train["shape_begin"] < val["shape_end"] and val["shape_begin"] < train["shape_end"]:
            raise ValueError("train and val shape seed ranges overlap")
    if abs(train["hg_g"] - val["hg_g"]) > 1e-6 or abs(train["albedo"] - val["albedo"]) > 1e-6:
        raise ValueError("train and val sets differ in hg_g/albedo")


class CloudNet(nn.Module):
    """13 -> 16 -> 16 -> 3, squared ReLU hidden activations, squared output (unclamped)."""

    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).clone())
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32).clone())
        self.l0 = nn.Linear(N_IN, N_HIDDEN)
        self.l1 = nn.Linear(N_HIDDEN, N_HIDDEN)
        self.l2 = nn.Linear(N_HIDDEN, N_OUT)

    def forward(self, desc):
        x = (desc - self.mean) * self.scale
        h = torch.square(torch.relu(self.l0(x)))
        h = torch.square(torch.relu(self.l1(h)))
        return torch.square(self.l2(h))

    def linears(self):
        return [self.l0, self.l1, self.l2]


@torch.no_grad()
def predict(model, x, clamp=None, batch=65536):
    outs = []
    for start in range(0, x.shape[0], batch):
        y = model(x[start:start + batch])
        if clamp is not None:
            y = torch.minimum(y, clamp)
        outs.append(y)
    return torch.cat(outs)


@torch.no_grad()
def mean_loss(model, x, y, batch=65536):
    total = torch.zeros((), device=x.device)
    for start in range(0, x.shape[0], batch):
        total += ((model(x[start:start + batch]) - y[start:start + batch]) ** 2).sum()
    return float(total) / (x.shape[0] * N_OUT)


def pearson(a, b):
    a = a.astype(np.float64) - a.mean()
    b = b.astype(np.float64) - b.mean()
    denom = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def rel_errors(a, b):
    rel = np.abs(a.astype(np.float64) - b) / (np.abs(b.astype(np.float64)) + 1e-3)
    return float(rel.max()), float(rel.mean())


def flat_weights(model):
    """W0[16][13], b0, W1[16][16], b1, W2[3][16], b2 as little-endian fp32 (CNR_NN_W0..B2)."""
    parts = []
    for layer in model.linears():
        parts.append(layer.weight.detach().cpu().numpy().astype(np.float32).reshape(-1))
        parts.append(layer.bias.detach().cpu().numpy().astype(np.float32).reshape(-1))
    weights = np.concatenate(parts).astype("<f4")
    if not np.isfinite(weights).all():
        raise ValueError("non-finite weights")
    if np.abs(weights).max() > 65504.0:
        raise ValueError("weights exceed the fp16 range used by shader_ml inference")
    return weights


def shader_ml_emulation(model, x, clamp):
    """numpy emulation of mlp_eval_shader_ml: fp16 weights, fp32 inputs/activations and accumulation."""
    def f16(v):
        return np.clip(v, -65504.0, 65504.0).astype(np.float16).astype(np.float32)

    mean = model.mean.cpu().numpy()
    scale = model.scale.cpu().numpy()
    layers = [(l.weight.detach().cpu().numpy().astype(np.float32), l.bias.detach().cpu().numpy().astype(np.float32))
              for l in model.linears()]
    h = ((x - mean) * scale).astype(np.float32)
    for i, (w, b) in enumerate(layers):
        z = h @ f16(w).T + b
        if i < len(layers) - 1:
            h = np.square(np.maximum(z, 0.0))
        else:
            h = np.minimum(np.square(z), clamp)
    return h.astype(np.float32)


def renderer_version():
    try:
        out = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=Path(__file__).resolve().parent,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip() or "nogit"
    except (OSError, subprocess.CalledProcessError):
        return "nogit"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--out", required=True, help="output prefix, e.g. models/cumulus_net")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=1, help="torch seed (init and shuffling)")
    parser.add_argument("--cpu", action="store_true", help="force CPU even if MPS is available")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or not args.lr > 0:
        parser.error("--epochs, --batch and --lr must be positive")

    # Output prefix; ".cnrm", ".json" and ".check.bin" are appended (names may contain dots).
    out = str(args.out)
    if out.endswith(".cnrm"):
        out = out[:-len(".cnrm")]
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    x_train, f_train, train_header = load_cnrd(args.train)
    x_val, f_val, val_header = load_cnrd(args.val)
    check_datasets(train_header, val_header)
    print(f"train {len(x_train)} records (shapes {train_header['shape_begin']}-{train_header['shape_end'] - 1}), "
          f"val {len(x_val)} records (shapes {val_header['shape_begin']}-{val_header['shape_end'] - 1})", flush=True)

    mean = x_train.astype(np.float64).mean(axis=0)
    std = np.maximum(x_train.astype(np.float64).std(axis=0), STD_FLOOR)
    scale = 1.0 / std
    clamp = np.maximum(CLAMP_FACTOR * f_train.astype(np.float64).max(axis=0), 1e-6).astype(np.float32)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() and not args.cpu else "cpu")
    xt = torch.from_numpy(x_train).to(device)
    yt = torch.from_numpy(f_train).to(device)
    xv = torch.from_numpy(x_val).to(device)
    yv = torch.from_numpy(f_val).to(device)

    model = CloudNet(mean.astype(np.float32), scale.astype(np.float32)).to(device)
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
            loss = nn.functional.mse_loss(model(xt[idx]), yt[idx])
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
    if best_state is None:
        raise RuntimeError("validation loss was never finite; nothing to export")

    best = CloudNet(mean.astype(np.float32), scale.astype(np.float32))
    best.load_state_dict(best_state)
    best.eval()
    clamp_t = torch.from_numpy(clamp)

    # Held-out metrics on the clamped fp32 outputs (what the renderer evaluates).
    x_val_t = torch.from_numpy(x_val)
    f_pred = predict(best, x_val_t, clamp_t).numpy()
    mse_per_output = ((f_pred.astype(np.float64) - f_val) ** 2).mean(axis=0)
    heldout = {
        "val_mse": float(mse_per_output.mean()),
        "val_mse_per_output": dict(zip(OUTPUT_NAMES, map(float, mse_per_output))),
        "pearson": dict(zip(OUTPUT_NAMES, (pearson(f_pred[:, i], f_val[:, i]) for i in range(N_OUT)))),
        "target_mean": dict(zip(OUTPUT_NAMES, map(float, f_val.astype(np.float64).mean(axis=0)))),
        "prediction_mean": dict(zip(OUTPUT_NAMES, map(float, f_pred.astype(np.float64).mean(axis=0)))),
    }
    print("held-out pearson " + " ".join(f"{k} {v:.4f}" for k, v in heldout["pearson"].items()), flush=True)

    weights = flat_weights(best)
    hg_g, albedo = train_header["hg_g"], train_header["albedo"]
    cnrm_path = Path(out + ".cnrm")
    cnrm = struct.pack(CNRM_FORMAT, b"CNRM", FORMAT_VERSION, N_IN, N_HIDDEN, N_HIDDEN_LAYERS, N_OUT, hg_g, albedo,
                       *RADII, *mean.astype(np.float32).tolist(), *scale.astype(np.float32).tolist(),
                       *clamp.tolist()) + weights.tobytes()
    cnrm_path.write_bytes(cnrm)

    # Parity data: fp32 PyTorch outputs (clamped) on the first validation records.
    count = min(CHECK_COUNT, len(x_val))
    x_check = np.ascontiguousarray(x_val[:count]).astype("<f4")
    f_check = predict(best, torch.from_numpy(x_check), clamp_t).numpy().astype("<f4")
    check_path = Path(out + ".check.bin")
    check_path.write_bytes(struct.pack(CNRC_FORMAT, b"CNRC", CHECK_VERSION, count, N_IN, N_OUT) +
                           x_check.tobytes() + f_check.tobytes())
    sim_max, sim_mean = rel_errors(shader_ml_emulation(best, x_check, clamp), f_check)

    def dataset_info(h):
        return {
            "path": h["path"], "sha256": h["sha256"], "version": h["version"], "n_records": h["n_records"],
            "shape_seed_range": [h["shape_begin"], h["shape_end"] - 1], "n_shapes": h["n_shapes"],
            "fixed_shape": bool(h["fixed_shape"]),
            "shape_ids": h["shape_ids"], "records_per_shape": h["records_per_shape"], "paths": h["paths"],
            "hg_g": h["hg_g"], "albedo": h["albedo"], "sigma_t_range": [h["sigma_t_min"], h["sigma_t_max"]],
        }

    sidecar = {
        "format_version": FORMAT_VERSION,
        "model": cnrm_path.name,
        "cnrm_sha256": hashlib.sha256(cnrm).hexdigest(),
        "function": "Ls(x,d) = f_sun*E_sun + f_sky*L_sky + f_gnd*L_gnd: full in-scattered radiance at x toward -d, "
                    "pre-albedo, including single-scattered sun",
        "architecture": {
            "n_in": N_IN, "n_hidden": N_HIDDEN, "n_hidden_layers": N_HIDDEN_LAYERS, "n_out": N_OUT,
            "inputs": INPUT_NAMES, "outputs": OUTPUT_NAMES, "radii_m": RADII,
            "activation": "squared relu max(z,0)^2", "output": "y = z^2, f = min(y, clamp)",
            "standardization": "x = (desc - mean) * scale, scale = 1/max(std, 1e-6), training-set statistics",
            "weights": "fp32, per layer W[out][in] row-major then bias[out]",
            "inference": {"scalar": "fp32 hand-written MSL",
                          "shader_ml": "matmul2d execution_thread, fp16 weights, fp32 inputs/activations/accumulation"},
        },
        "standardization": {"mean": mean.astype(np.float32).tolist(), "scale": scale.astype(np.float32).tolist(),
                            "std_floor": STD_FLOOR},
        "clamp": dict(zip(OUTPUT_NAMES, map(float, clamp))),
        "clamp_rule": f"{CLAMP_FACTOR} x max training target per output",
        "medium": {"hg_g": hg_g, "albedo": albedo},
        "hyperparameters": {
            "optimizer": "adam", "lr": args.lr, "schedule": "cosine decay per step", "batch": args.batch,
            "epochs": args.epochs, "loss": "mse(y, f) on raw targets", "torch_seed": args.seed,
            "device": str(device),
        },
        "datasets": {"train": dataset_info(train_header), "val": dataset_info(val_header)},
        "train_seed_range": [train_header["shape_begin"], train_header["shape_end"] - 1],
        "val_seed_range": [val_header["shape_begin"], val_header["shape_end"] - 1],
        "versions": {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
                     "cnrd": CNRD_VERSION, "cnrm": FORMAT_VERSION, "cnrc": CHECK_VERSION},
        "renderer_version": renderer_version(),
        "loss_curves": {"train": train_curve, "val": val_curve},
        "best_epoch": best_epoch + 1,
        "best_val_loss": best_val,
        "heldout": heldout,
        "check_file": check_path.name,
        "check_count": count,
        "shader_ml_fp16_emulation_rel_error": {"max": sim_max, "mean": sim_mean},
        "train_seconds": time.time() - start,
        "known_bias": f"targets are {train_header['paths']}-path Monte Carlo estimates; MSE on raw targets is "
                      "unbiased by zero-mean target noise but the clamp and finite capacity are not",
    }
    json_path = Path(out + ".json")
    json_path.write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"wrote {cnrm_path} ({len(weights)} fp32 weights), {json_path}, {check_path}; best epoch "
          f"{best_epoch + 1} val {best_val:.6g}; shader_ml fp16 emulation max rel {sim_max:.3g}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
