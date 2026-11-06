#!/usr/bin/env python3
"""Export a .cnrm v2 model to a Metal ML package for MTL4MachineLearningCommandEncoder inference.

Pipeline: models/<name>.cnrm -> torch network (the inference math: standardization, 13 -> 16 -> 16 -> 3
with squared ReLU, squared output, min(y, clamp)) -> parity check against models/<name>.check.bin ->
Core ML mlprogram (fp32, flexible batch: input "x" [N, 13], output "f" [N, 3]) models/<name>.mlpackage ->
metal-package-builder -> models/<name>.mtlpackage, plus models/<name>.mtlpackage.json (provenance).

The standardization is part of the exported network, so the renderer feeds RAW descriptors.

metal-package-builder (Xcode Metal toolchain) lives in a read-only cryptex mount and looks for
<dir containing Metal.xctoolchain>/XcodeDefault.xctoolchain/usr/bin/coremlcompiler, which does not exist
there. The script runs the unmodified binary from a temporary directory that recreates that layout with a
copy of the binary and symlinks.

Usage:
    .venv/bin/python tools/export_mtlpackage.py --model models/cumulus_single.cnrm
"""

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import (CNRC_FORMAT, CNRM_FORMAT, FORMAT_VERSION, CHECK_VERSION, N_HIDDEN, N_HIDDEN_LAYERS, N_IN,  # noqa: E402
                   N_OUT, RADII, CloudNet, rel_errors, renderer_version)

# Largest batch (rows) the flexible input accepts; src/neural_mlencoder.h kMLEncoderMaxBatch must match.
MAX_BATCH = 1 << 24
TORCH_PARITY_TOLERANCE = 1e-5
MTARGET_OS = "macos26.0"


class ClampedCloudNet(nn.Module):
    """CloudNet followed by the inference clamp f = min(y, clamp)."""

    def __init__(self, net, clamp):
        super().__init__()
        self.net = net
        self.register_buffer("clamp", torch.as_tensor(clamp, dtype=torch.float32).clone())

    def forward(self, desc):
        return torch.minimum(self.net(desc), self.clamp)


def load_cnrm(path):
    data = Path(path).read_bytes()
    header_size = struct.calcsize(CNRM_FORMAT)
    if len(data) < header_size:
        raise ValueError(f"{path}: too small for a .cnrm header")
    h = struct.unpack_from(CNRM_FORMAT, data, 0)
    magic, version, n_in, n_hidden, n_hidden_layers, n_out = h[:6]
    if magic != b"CNRM" or version != FORMAT_VERSION:
        raise ValueError(f"{path}: not a CNRM format_version {FORMAT_VERSION} file")
    if (n_in, n_hidden, n_hidden_layers, n_out) != (N_IN, N_HIDDEN, N_HIDDEN_LAYERS, N_OUT):
        raise ValueError(f"{path}: dimensions ({n_in}, {n_hidden}, {n_hidden_layers}, {n_out}) are not "
                         f"({N_IN}, {N_HIDDEN}, {N_HIDDEN_LAYERS}, {N_OUT})")
    hg_g, albedo = h[6], h[7]
    radii = list(h[8:14])
    if any(abs(a - b) > 1e-6 * b for a, b in zip(radii, RADII)):
        raise ValueError(f"{path}: descriptor radii {radii} differ from {RADII}")
    mean = np.array(h[14:27], dtype=np.float32)
    scale = np.array(h[27:40], dtype=np.float32)
    clamp = np.array(h[40:43], dtype=np.float32)
    shapes = [(N_HIDDEN, N_IN), (N_HIDDEN, N_HIDDEN), (N_OUT, N_HIDDEN)]
    count = sum(o * i + o for o, i in shapes)
    if len(data) != header_size + 4 * count:
        raise ValueError(f"{path}: size {len(data)} bytes, expected {header_size + 4 * count}")
    weights = np.frombuffer(data, dtype="<f4", count=count, offset=header_size).astype(np.float32)

    net = CloudNet(mean, scale)
    offset = 0
    with torch.no_grad():
        for layer, (o, i) in zip(net.linears(), shapes):
            layer.weight.copy_(torch.from_numpy(weights[offset:offset + o * i].reshape(o, i)))
            offset += o * i
            layer.bias.copy_(torch.from_numpy(weights[offset:offset + o]))
            offset += o
    model = ClampedCloudNet(net, clamp).eval()
    return model, {"hg_g": hg_g, "albedo": albedo, "sha256": hashlib.sha256(data).hexdigest()}


def load_check(path):
    data = Path(path).read_bytes()
    header_size = struct.calcsize(CNRC_FORMAT)
    magic, version, count, n_in, n_out = struct.unpack_from(CNRC_FORMAT, data, 0)
    if magic != b"CNRC" or version != CHECK_VERSION or n_in != N_IN or n_out != N_OUT:
        raise ValueError(f"{path}: not a CNRC version {CHECK_VERSION} file for {N_IN} -> {N_OUT}")
    if len(data) != header_size + 4 * count * (n_in + n_out):
        raise ValueError(f"{path}: size does not match its header")
    x = np.frombuffer(data, dtype="<f4", count=count * n_in, offset=header_size).reshape(count, n_in)
    f = np.frombuffer(data, dtype="<f4", count=count * n_out, offset=header_size + 4 * count * n_in)
    return x.astype(np.float32), f.reshape(count, n_out).astype(np.float32)


def xcrun_find(tool):
    try:
        out = subprocess.run(["xcrun", "--find", tool], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as e:
        stderr = getattr(e, "stderr", "") or str(e)
        raise RuntimeError(f"`xcrun --find {tool}` failed ({stderr.strip()}). The Xcode Metal toolchain is required: "
                           "accept the Xcode license, install the Metal toolchain (xcodebuild "
                           "-downloadComponent MetalToolchain) and point xcode-select or DEVELOPER_DIR at Xcode.")
    path = Path(out.stdout.strip())
    if not path.is_file():
        raise RuntimeError(f"`xcrun --find {tool}` returned {path}, which does not exist")
    return path


def build_mtlpackage(mlpackage, mtlpackage):
    """Runs metal-package-builder from a temporary toolchain layout; returns the builder path used."""
    builder = xcrun_find("metal-package-builder")
    coremlcompiler = xcrun_find("coremlcompiler")
    toolchain = builder.parents[2]  # .../Metal.xctoolchain (builder is usr/bin/metal-package-builder)
    if toolchain.name != "Metal.xctoolchain":
        raise RuntimeError(f"unexpected metal-package-builder location {builder} (expected "
                           ".../Metal.xctoolchain/usr/bin/metal-package-builder)")
    with tempfile.TemporaryDirectory(prefix="cloudnr_mpb_") as tmp:
        root = Path(tmp)
        bin_dir = root / "Metal.xctoolchain" / "usr" / "bin"
        bin_dir.mkdir(parents=True)
        local_builder = bin_dir / "metal-package-builder"
        shutil.copy2(builder, local_builder)
        (root / "Metal.xctoolchain" / "usr" / "lib").symlink_to(toolchain / "usr" / "lib")
        if (toolchain / "System").exists():
            (root / "Metal.xctoolchain" / "System").symlink_to(toolchain / "System")
        default_bin = root / "XcodeDefault.xctoolchain" / "usr" / "bin"
        default_bin.mkdir(parents=True)
        (default_bin / "coremlcompiler").symlink_to(coremlcompiler)

        if mtlpackage.exists():
            shutil.rmtree(mtlpackage)
        cmd = [str(local_builder), str(mlpackage), "-ml", "-o", str(mtlpackage), "--mtargetos", MTARGET_OS]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not mtlpackage.exists():
            raise RuntimeError(f"metal-package-builder failed (exit {result.returncode}):\n{result.stdout}{result.stderr}")
    return builder


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="models/<name>.cnrm")
    parser.add_argument("--check", help="CNRC v2 parity file (default: <name>.check.bin next to the model)")
    args = parser.parse_args()

    import coremltools as ct

    cnrm = Path(args.model)
    if cnrm.suffix != ".cnrm":
        raise SystemExit(f"{cnrm}: expected a .cnrm file")
    stem = cnrm.with_suffix("")
    check = Path(args.check) if args.check else Path(str(stem) + ".check.bin")
    mlpackage = Path(str(stem) + ".mlpackage")
    mtlpackage = Path(str(stem) + ".mtlpackage")
    sidecar = Path(str(stem) + ".mtlpackage.json")

    model, info = load_cnrm(cnrm)
    x_check, f_check = load_check(check)
    with torch.no_grad():
        f_torch = model(torch.from_numpy(x_check)).numpy()
    parity_max, parity_mean = rel_errors(f_torch, f_check)
    print(f"torch rebuild vs {check}: max rel {parity_max:.3g}, mean rel {parity_mean:.3g}", flush=True)
    if not np.isfinite(f_torch).all() or parity_max >= TORCH_PARITY_TOLERANCE:
        raise SystemExit(f"rebuilt network does not reproduce {check} (max rel {parity_max:.3g} >= "
                         f"{TORCH_PARITY_TOLERANCE})")

    traced = torch.jit.trace(model, torch.zeros(64, N_IN, dtype=torch.float32))
    mlmodel = ct.convert(traced,
                         inputs=[ct.TensorType(name="x", shape=(ct.RangeDim(1, MAX_BATCH, default=64), N_IN))],
                         outputs=[ct.TensorType(name="f")],
                         convert_to="mlprogram",
                         compute_precision=ct.precision.FLOAT32,
                         # metal-package-builder 27.2 has no rewrite patterns for the ios19 (macOS 26) opset's
                         # binary elementwise ops (sub/mul/minimum abort with "Missing pattern to rewrite MIL op");
                         # the ios18 (macOS 15) opset lowers fully.
                         minimum_deployment_target=ct.target.macOS15)
    if mlpackage.exists():
        shutil.rmtree(mlpackage)
    mlmodel.save(str(mlpackage))
    print(f"wrote {mlpackage}", flush=True)

    builder = build_mtlpackage(mlpackage, mtlpackage)
    print(f"wrote {mtlpackage}", flush=True)

    meta = {
        "source_model": str(cnrm),
        "source_sha256": info["sha256"],
        "hg_g": info["hg_g"],
        "albedo": info["albedo"],
        "mlpackage": str(mlpackage),
        "mtlpackage": str(mtlpackage),
        "function": "main",
        "inputs": {"x": {"shape": ["N", N_IN], "dtype": "float32", "content": "raw descriptor (standardization "
                                                                              "is inside the network)"}},
        "outputs": {"f": {"shape": ["N", N_OUT], "dtype": "float32", "content": "min(y, clamp) = (f_sun, f_sky, f_gnd)"}},
        "max_batch": MAX_BATCH,
        "precision": "fp32",
        "torch_parity": {"check_file": str(check), "max_rel": parity_max, "mean_rel": parity_mean},
        "coremltools_version": ct.__version__,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "builder": str(builder),
        "builder_args": ["-ml", "--mtargetos", MTARGET_OS],
        "renderer_version": renderer_version(),
    }
    sidecar.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {sidecar}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
