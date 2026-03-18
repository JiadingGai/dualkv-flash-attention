# Installation Guide for dualkv-flash-attention

## Prerequisites

| Requirement | Version used | Notes |
|---|---|---|
| Python | 3.12 | via virtualenv |
| PyTorch | 2.10.0+cu128 | must be installed before building |
| CUDA toolkit | 12.9 | `nvcc` must be on `PATH` |
| GPU | A100 (sm_80) | adjust `FLASH_ATTN_CUDA_ARCHS` for other GPUs |

---

## Validated stack

This branch has been validated with the following downstream stack for rollout
generation in RL training:

| Component | Version |
|---|---|
| dualkv-flash-attention branch | `gai_debug_flash_decoding_gxpo` |
| vLLM | `v0.8.5` |
| VERL | `v0.7.0` |

Important integration note: the current DualKV kernel path expects separate
context and decoded KV caches. vLLM `v0.8.5` still uses a single paged KV cache
layout in its FlashAttention backend, so a custom interface layer on top of
vLLM is required to route decode attention into `flash_attn_with_kvcache(...,
use_dualkv_attention=True)`.

---

## One-time system setup

Install the Python development headers (required for compiling the C++ extension):

```sh
sudo apt-get install -y python3.12-dev
```

Install build tools into your virtualenv:

```sh
pip install ninja psutil packaging
```

---

## Editable install (development mode)

```sh
cd dualkv-flash-attention
CUTLASS_ROOT_DIR=$(readlink -f csrc/cutlass) pip install -e . --no-build-isolation
```

### Environment variables

| Variable | Value | Purpose |
|---|---|---|
| `CUTLASS_ROOT_DIR` | `$(readlink -f csrc/cutlass)` | Points to the bundled Cutlass submodule |
| `FLASH_ATTN_CUDA_ARCHS` | `"80"` *(optional)* | Compile only for A100 (sm_80). Omit to compile for all supported archs (80;90;100;120), which takes much longer |
| `MAX_JOBS` | e.g. `32` *(optional)* | Ninja parallel jobs. Default is auto-detected from CPU cores and free RAM |

### Full example with optional speedup flags

```sh
CUTLASS_ROOT_DIR=$(readlink -f csrc/cutlass) \
FLASH_ATTN_CUDA_ARCHS="80" \
MAX_JOBS=32 \
pip install -e . --no-build-isolation
```

---

## Why `--no-build-isolation`?

pip has two build modes:

- **Legacy** (no `pyproject.toml`): calls `setup.py` directly in the current Python environment - sees all installed packages including torch.
- **PEP 517** (with `pyproject.toml`): by default creates an *isolated* build environment containing only the packages listed in `[build-system] requires`. torch is NOT in that list (it's too large), so `setup.py` crashes with `ModuleNotFoundError: No module named 'torch'`.

`--no-build-isolation` disables the isolated env and uses the current virtualenv as-is, where torch is already installed. This is the standard approach for GPU-accelerated packages that depend on torch at build time.

---

## Why the `pyproject.toml` at the repo root was needed

Without a `pyproject.toml`, `pip install -e .` fell back to `setup.py develop`. Modern versions of setuptools' `develop` command internally re-invoke pip as a subprocess with `--use-pep517`, creating a fresh isolated env - which again lacks torch. This caused an infinite failure loop.

Adding a minimal `pyproject.toml` at the repo root:

```toml
# pyproject.toml
[build-system]
requires = ["setuptools", "wheel", "packaging", "ninja", "psutil"]
build-backend = "setuptools.build_meta"
```

tells pip to use the PEP 517 path from the start, which properly respects `--no-build-isolation`.

---

## Verifying the install

```python
import flash_attn
import flash_attn_2_cuda
from flash_attn.flash_attn_interface import flash_attn_with_kvcache

print(flash_attn.__version__)                      # 2.7.4.post1
print(flash_attn.__file__)                         # .../dualkv-flash-attention/flash_attn/__init__.py
print(hasattr(flash_attn_2_cuda, 'fwd_kvcache_dualkv'))  # True
```

The `flash_attn.__file__` pointing into the source tree confirms it is editable (live source).

---

## Rebuild after CUDA source changes

If you modify any `.cu` or `.h` files, rerun:

```sh
CUTLASS_ROOT_DIR=$(readlink -f csrc/cutlass) pip install -e . --no-build-isolation
```

Ninja caches compiled `.o` files, so only changed translation units are recompiled.
