# Linux + AMD ROCm (gfx1030 / RDNA2)

This branch runs FreeToken on Linux with AMD ROCm, targeting the Radeon Pro
V620 (gfx1030, RDNA2). It has been verified end-to-end **only on that one
card** — see the [status](#status) section for exactly what was tested.

Why gfx1030 is its own chapter: RDNA2 differs from the RDNA3/RDMA4 targets of
the Windows port in ways that leak into kernels — LLVM lowers bf16
multiply-accumulates to `fdot2` (missing on gfx1030), the hipcc-built ggml
extension runs wave64, and int32 offsets in the GDN state kernels overflow at
27B-class pool sizes. All of these are fixed on this branch; none change
results on other GPUs.

## Requirements

- Linux x86_64 (tested on CachyOS, kernel 7.2.2)
- AMD RDNA2+ GPU. gfx1030 verified; gfx1101/gfx1200/gfx1201 are covered by the
  Windows port and should work with the arch strings swapped
- 32 GB system RAM minimum (64 GB recommended)
- Python 3.12 (the ROCm nightly wheels are cp312)

## Install

```bash
git clone <this repo> && cd FreeToken-rocm-test
python3.12 -m venv .venv 2>/dev/null || uv venv --python 3.12 .venv
source .venv/bin/activate
export PIP_WHEEL_DIR=  # unused, see pinned wheels below

# 1) matched ROCm nightly stack (torch and rocm-sdk MUST share the stamp)
uv pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ --no-deps \
    "torch==2.11.0+rocm10.1.0a20260819" \
    "amd-torch-device-gfx1030==2.11.0+rocm10.1.0a20260819" \
    "rocm-sdk-core==10.1.0a20260819" "rocm-sdk-libraries==10.1.0a20260819" \
    "rocm-sdk-devel==10.1.0a20260819" "rocm-sdk-device-gfx1030==10.1.0a20260819" \
    "triton==3.7.1+git0263a6a6.rocm7.15.0a20260712"
uv pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
    --no-deps --no-build-isolation "rocm==10.1.0a20260819"

# 2) FreeToken without the CUDA extensions, plus its runtime deps
uv pip install -e ".[accel]" --no-deps --no-build-isolation
uv pip install einops fastapi "flashlib==0.3.0" gguf "huggingface_hub>=1.5" msgpack \
    modelscope "numpy>=2.0,<2.5" openai partial-json-parser prompt_toolkit \
    "pydantic>=2.9" "pyzmq>=27" safetensors tqdm "transformers>=5.5" uvicorn \
    "apache-tvm-ffi==0.1.13.post3" typing_extensions filelock sympy networkx \
    jinja2 fsspec setuptools

# 3) make triton import before torch (avoids a lazy-import deadlock, see
#    sitecustomize.py note in docs below)
```

Then source the environment (see `linux/freetoken-env.sh`) and verify:

```bash
source linux/freetoken-env.sh
python -c "import torch; print(torch.cuda.get_device_name(0))"
ft serve --model <your-model.gguf> --host 0.0.0.0 --port 8185
```

## Environment

`linux/freetoken-env.sh` sets every variable the stack needs; the important
ones and why:

| Variable | Why |
|---|---|
| `HIP_PATH` / `ROCM_HOME` → venv SDK | the venv's matched nightly SDK must win over any system ROCm (`/opt/rocm`) |
| `TVM_FFI_ROCM_ARCH_LIST=gfx1030` | without it tvm-ffi JIT compiles for gfx906 → dead kernels |
| `PYTORCH_ROCM_ARCH`, `TRITON_OVERRIDE_ARCH` | codegen target for JIT builds |
| `HIP_DEVICE_LIB_PATH` | bitcode location for hipcc JIT compiles |
| `PYTORCH_ALLOC_CONF=expandable_segments:True` | the engine sets this itself; keep it |
| `sitecustomize.py` in the venv | imports triton before torch — importing torch first deadlocks libtriton's init in this stack |

## Serving a model as a systemd service

See `linux/freetoken-qwen38-27b.user-service.example` — a user-level unit
(`systemctl --user start freetoken-qwen38-27b`) that loads
`Unsloth-Qwen3.8-27B-UD-IQ4_XS.gguf` (14 GB) fully in VRAM, opens an
OpenAI-compatible API on :8185, and pins GPU clocks while running. Edit the
paths for your machine.

## Status

Verified end-to-end on a single Radeon Pro V620 (gfx1030, 32 GB):

- Model: `Unsloth-Qwen3.8-27B-UD-IQ4_XS.gguf` (Qwen3.8-27B, hybrid GDN +
  full-attention, 64 layers, IQ4_XS/Q5_K/Q6_K/Q8_0 mix)
- Output verified against llama.cpp (same top-5 ranking, matching logit gaps)
  and against the HF safetensors checkpoint for every GDN tensor
- Chat with thinking (`reasoning_content` + `content`), 262144 ctx,
  ~9.8 tok/s decode (first untuned measurement)
- `Qwen3.5-0.8B.gguf` (bf16, nv == nk) as a regression check

Not yet tested: MoE checkpoints (`qwen35moe` path), multi-GPU, RDNA3/4 on
Linux, iGPUs.

## Known limitations

- Speculative decoding / draft models: not implemented in FreeToken (any backend).
- Vision (mmproj): FreeToken serves text-only.
- No warranty. See [STATUS](#status): one GPU, one model file.
