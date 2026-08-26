"""Exact-size pinned host tensors (e.g. offload expert banks).

The offload gather kernel (``fast_index_copy``) reads host memory zero-copy from the
GPU, so allocations must be pinned + device-mapped. We avoid
``torch.empty(pin_memory=True)`` because its caching allocator rounds sizes up to the
next power of two (a 70GB bank would reserve 128GB)."""

from __future__ import annotations

import importlib
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _load_pinned_extension():
    try:
        return importlib.import_module("freetoken.kernel._pinned_tensor")
    except ImportError:
        # patched: ROCm/Windows fallback -- no exact-size pinned extension;
        # callers fall back to torch's own pinned allocator (rounds sizes up).
        return None


def create_pinned_tensor_like(input: torch.Tensor) -> torch.Tensor:
    """Create a CPU pinned tensor with the same size, stride, and dtype as input."""
    ext = _load_pinned_extension()
    if ext is None:
        out = torch.empty_like(input, pin_memory=True)
        return out
    return ext.create_pinned_tensor_like(input)


def copy_to_pinned_tensor(input: torch.Tensor) -> torch.Tensor:
    """Copy a CPU tensor into exact-size cudaMallocHost pinned storage."""

    output = create_pinned_tensor_like(input)
    with torch.no_grad():
        output.copy_(input)
    return output


def alloc_pinned_tensor(*shape: int, dtype: torch.dtype) -> torch.Tensor:
    """Allocate an exact-size, uninitialized pinned host tensor via cudaHostAlloc."""
    ext = _load_pinned_extension()
    if ext is None:
        return torch.empty(*shape, dtype=dtype, pin_memory=True)
    return ext.alloc_pinned_tensor(list(shape), dtype)


@lru_cache(maxsize=1)
def _hip_runtime():
    """ctypes handle to the HIP runtime, for hipHostRegister/hipHostGetDevicePointer
    when the ``_pinned_tensor`` extension is absent (the ROCm/Windows port never
    builds it). The DLL is already loaded by torch, so CDLL by name resolves it."""
    if getattr(torch.version, "hip", None) is None:
        return None
    import ctypes

    for name in ("amdhip64_7.dll", "amdhip64.dll", "libamdhip64.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def host_register(addr: int, nbytes: int) -> None:
    """cudaHostRegister ``nbytes`` at ``addr`` as portable+mapped (pin-after-fill).

    Without the extension this used to be a SILENT NO-OP, which left the expert
    banks pageable -- the fused offload gather then dereferences unregistered host
    memory from the GPU and dies with `unspecified launch failure` (the RDNA4
    offload-decode TDR). On ROCm, register through the HIP runtime instead."""
    ext = _load_pinned_extension()
    if ext is not None:
        ext.host_register(addr, nbytes)
        return
    hip = _hip_runtime()
    if hip is not None:
        import ctypes

        # hipHostRegisterPortable (1) | hipHostRegisterMapped (2)
        status = hip.hipHostRegister(
            ctypes.c_void_p(addr), ctypes.c_size_t(nbytes), ctypes.c_uint(3)
        )
        if status != 0:
            raise RuntimeError(
                f"hipHostRegister({nbytes} bytes) failed with hipError {status}"
            )
        return
    # CUDA without the extension: keep the historical no-op (that path always ran
    # with the extension built); the GPU-deref consumers are gated off it anyway.


@lru_cache(maxsize=1)
def _host_ptr_identity() -> bool:
    # cached per process: FreeToken pins one CUDA device per process (set at engine launch)
    ext = _load_pinned_extension()
    if ext is not None:
        return bool(ext.host_ptr_identity())
    hip = _hip_runtime()
    if hip is None:
        return False
    import ctypes
    import mmap

    # Probe with hipHostREGISTERed memory -- how the expert banks are pinned. On
    # Windows/WDDM+ROCm, registered memory maps to a DIFFERENT device address even
    # though hipHostMalloc'd memory (torch pin_memory) is unified, so probing a
    # pin_memory tensor here would wrongly report identity and hand the GPU host
    # VAs (the offload-decode `unspecified launch failure`).
    buf = mmap.mmap(-1, 4096)
    addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    if hip.hipHostRegister(
        ctypes.c_void_p(addr), ctypes.c_size_t(4096), ctypes.c_uint(3)
    ) != 0:
        return False
    dev = ctypes.c_void_p()
    ok = (
        hip.hipHostGetDevicePointer(
            ctypes.byref(dev), ctypes.c_void_p(addr), ctypes.c_uint(0)
        )
        == 0
    )
    identity = ok and dev.value == addr
    hip.hipHostUnregister(ctypes.c_void_p(addr))
    return identity


def device_ptr(t: torch.Tensor) -> int:
    """Base address of ``t`` as the GPU must dereference it.

    Equals ``data_ptr()`` on CUDA tensors and wherever pinned host memory is
    device-visible at its host VA (Linux/UVA, and ROCm/Windows per the probe).
    Where registered memory maps to a different device address, zero-copy consumers
    must use this, not ``data_ptr()``. Host tensors must be pinned+mapped."""
    if t.is_cuda or _host_ptr_identity():
        return t.data_ptr()
    ext = _load_pinned_extension()
    if ext is not None:
        return ext.host_device_ptr(t.data_ptr())
    hip = _hip_runtime()
    if hip is not None:
        import ctypes

        dev = ctypes.c_void_p()
        status = hip.hipHostGetDevicePointer(
            ctypes.byref(dev), ctypes.c_void_p(t.data_ptr()), ctypes.c_uint(0)
        )
        if status != 0:
            raise RuntimeError(f"hipHostGetDevicePointer failed with hipError {status}")
        return int(dev.value)
    return t.data_ptr()
