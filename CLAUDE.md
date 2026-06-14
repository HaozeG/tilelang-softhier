# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is TileLang

TileLang is a Python-based DSL for writing high-performance GPU/CPU kernels (GEMM, FlashAttention, dequant GEMM, etc.). Users write kernels in Python decorated with `@tilelang.jit`, which compiles them through TVM's IR to CUDA/HIP/Metal. The project has both a Python layer (`tilelang/`) and a C++ layer (`src/`) built with CMake/scikit-build-core.

## Build Commands

```bash
# Install from source (development mode)
pip install -e . -v

# Manual CMake build
mkdir -p build && cd build
cmake .. -DUSE_CUDA=ON   # or -DUSE_ROCM=ON, -DUSE_METAL=ON
make -j$(nproc)
export PYTHONPATH=/path/to/repo:$PYTHONPATH
```

## Test Commands

```bash
# Run all tests
python -m pytest testing

# Run a specific test file
python -m pytest testing/python/test_gemm.py

# Run on a specific GPU
CUDA_VISIBLE_DEVICES=0 python -m pytest testing
```

Tests live in `testing/python/`. The `conftest.py` ensures the in-tree package is imported (not a globally installed one) and seeds random state for reproducibility.

## Linting and Formatting

```bash
# Run all lint checks
pre-commit run --all-files

# Auto-fix Python issues
ruff check --fix tilelang/
ruff format tilelang/

# C++ formatting
clang-format -i src/**/*.cc src/**/*.h
```

Tools: **ruff** (Python, line length 140), **clang-format** (C++), **codespell** (spelling). Config is in `pyproject.toml` and `.pre-commit-config.yaml`. The `3rdparty/`, `build/`, and `examples/` directories are excluded.

## Architecture Overview

### Compilation Pipeline

User Python kernel → `tilelang.jit` → TVM Script parsing → TVM IR (TIR) → `engine/lower.py` lowering passes → C++/CUDA codegen → compiled binary.

Key stages:
1. **Parsing** (`tilelang/language/parser/`): TVM Script parser extended with TileLang constructs.
2. **Lowering** (`tilelang/engine/lower.py`, `engine/phase.py`): Applies a sequence of TVM and custom IR transformation passes.
3. **Transforms** (`tilelang/transform/`, `src/transform/`): Python and C++ IR passes (vectorization, shared memory layout, pipeline scheduling, fence injection, etc.).
4. **Codegen** (`src/target/`): CUDA (`codegen_cuda.cc`), HIP, Metal, CuTeDSL backends.
5. **JIT execution** (`tilelang/jit/`): Wraps the compiled binary; supports multiple execution backends (TVM FFI, Cython, NVRTC, Torch).

### Key Python Modules

| Module | Purpose |
|--------|---------|
| `tilelang/jit/` | `compile()`, `@jit` decorator, `JITKernel`, execution backends |
| `tilelang/language/` | All DSL constructs: `T.Kernel`, `T.copy`, `T.gemm`, `T.alloc_*`, loop types |
| `tilelang/engine/` | Lowering pipeline, compiler phases, post-processing callbacks |
| `tilelang/layout/` | `Layout` and `Fragment` memory layout abstractions |
| `tilelang/transform/` | Python-side IR transformation passes |
| `tilelang/analysis/` | Program analysis passes used during compilation |
| `tilelang/autotuner/` | Auto-tuning infrastructure |
| `tilelang/intrinsics/` | Low-level hardware intrinsic mappings |
| `tilelang/tileop/` | Tile-level operator implementations (GEMM, sparse GEMM) |
| `tilelang/carver/arch/` | Architecture-specific capabilities (compute capability, register counts) |

### Key C++ Source (`src/`)

- `src/transform/` — C++ IR passes (vectorization, layout rewrite, software pipelining, PTX generation)
- `src/target/` — Code generation backends; `ptx.cc` for inline PTX, `codegen_cuda.cc`, `codegen_cutedsl.cc`
- `src/tl_templates/` — CUTLASS/Composable Kernel template headers used in generated code
- `src/layout/` — C++ layout transformation logic

### User-Facing API Pattern

```python
import tilelang as tl
import tilelang.language as T

@tl.jit
def matmul(A: T.Tensor[(M, K), "float16"], B: T.Tensor[(K, N), "float16"],
           C: T.Tensor[(M, N), "float32"]):
    with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), "float16")
        B_shared = T.alloc_shared((block_K, block_N), "float16")
        C_frag   = T.alloc_fragment((block_M, block_N), "float32")
        T.clear(C_frag)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[bx * block_M:, k * block_K:], A_shared)
            T.copy(B[k * block_K:, by * block_N:], B_shared)
            T.gemm(A_shared, B_shared, C_frag)
        T.copy(C_frag, C[bx * block_M:, by * block_N:])

result = matmul(a, b)                       # auto-compiles and runs
src    = matmul.get_kernel_source()         # inspect generated CUDA
perf   = matmul.get_profiler().do_bench()   # benchmark
```

### Multi-Backend Support

- **CUDA** (NVIDIA): default, uses CUTLASS templates
- **HIP/ROCm** (AMD): via `-DUSE_ROCM=ON`, uses Composable Kernel templates
- **Metal** (Apple): via `-DUSE_METAL=ON`, additional transforms in `tilelang/transform/metal/`
- **CuTeDSL**: experimental backend via `src/target/codegen_cutedsl.cc`

The target is auto-detected from the environment but can be overridden via `compile(func, target="cuda")`.
