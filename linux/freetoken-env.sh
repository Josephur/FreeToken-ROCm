#!/usr/bin/env bash
# FreeToken-ROCm (Linux) runtime environment.
# Set FT_ROOT to the directory containing the FreeToken-rocm-test checkout and
# FT_MODEL_DIR to wherever your GGUF models live, then: source this file.
#
#   export FT_ROOT=~/FreeToken-rocm-test
#   export FT_MODEL_DIR=~/models
#   source "$FT_ROOT/linux/freetoken-env.sh"

FT_ROOT="${FT_ROOT:-$HOME/FreeToken-rocm-test}"
PYVER=python3.12
FT_VENV="$FT_ROOT/.venv"
SITE="$FT_VENV/lib/$PYVER/site-packages"

HIP_PATH="$SITE/_rocm_sdk_core"
export HIP_PATH ROCM_HOME="$HIP_PATH" ROCM_PATH="$HIP_PATH"
export PATH="$HIP_PATH/bin:$FT_VENV/bin:$PATH"
export HIP_DEVICE_LIB_PATH="$HIP_PATH/lib/llvm/amdgcn/bitcode"
export TVM_FFI_ROCM_ARCH_LIST="${TVM_FFI_ROCM_ARCH_LIST:-gfx1030}"
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx1030}"
export TRITON_OVERRIDE_ARCH="${TRITON_OVERRIDE_ARCH:-gfx1030}"
export ROCM_SDK_TARGET_FAMILY="${ROCM_SDK_TARGET_FAMILY:-gfx1030}"
export TVM_FFI_CACHE_DIR="${TVM_FFI_CACHE_DIR:-$FT_ROOT/.tvm-ffi-cache}"
export CC="$HIP_PATH/lib/llvm/bin/clang"
export CXX="$HIP_PATH/lib/llvm/bin/clang++"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$TVM_FFI_CACHE_DIR"
