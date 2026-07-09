# CUDA dtype support matrix

This matrix records CUDA dtype names visible in the local CUDA headers and how
Vectra currently maps them through the optional Axiom accelerator bridge.  It is
an integration status document, not a claim that native CUDA kernels exist for
all listed dtypes.

Local CUDA evidence: `/usr/local/cuda/include/library_types.h` declares
`cudaDataType_t` values including:

| CUDA dtype | Meaning in CUDA headers | Vectra dtype | Axiom bridge status |
| --- | --- | --- | --- |
| `CUDA_R_16F` | real half | `f16` | Supported for same-shape add/sub/mul/div and contiguous 2D matmul through widen-to-f32 Axiom CUDA runtime seed, then narrowed to f16. Native f16 CUDA lowering remains future work. |
| `CUDA_C_16F` | complex half pair | Not exposed | Planned. |
| `CUDA_R_16BF` | real bfloat16 | `BFloat16` | Supported for same-shape add/sub/mul/div and contiguous 2D matmul through widen-to-f32 Axiom CUDA runtime seed, then narrowed to BFloat16. Native BF16/CUTILE tensor-core lowering remains future work. |
| `CUDA_C_16BF` | complex bfloat16 pair | Not exposed | Planned. |
| `CUDA_R_32F` | real float | `f32` | Native Axiom CUDA seed for add/sub/mul/div, SAXPY, scalar materialization, and CUDA Tile IR matmul. |
| `CUDA_C_32F` | complex float pair | `Complex64` | CPU Vectra support exists; Axiom CUDA bridge planned. |
| `CUDA_R_64F` | real double | `f64` | Axiom CPU→Veyra path for supported ops; CUDA bridge planned. |
| `CUDA_C_64F` | complex double pair | `Complex128` | CPU Vectra support exists; Axiom CUDA bridge planned. |
| `CUDA_R_4I` / `CUDA_C_4I` | signed 4-bit integer / pair | Not exposed | Planned packed dtype. |
| `CUDA_R_4U` / `CUDA_C_4U` | unsigned 4-bit integer / pair | Not exposed | Planned packed dtype. |
| `CUDA_R_8I` / `CUDA_C_8I` | signed 8-bit integer / pair | `i8` / not exposed complex pair | CPU Vectra dtype exists; Axiom CUDA bridge planned. |
| `CUDA_R_8U` / `CUDA_C_8U` | unsigned 8-bit integer / pair | `u8` / not exposed complex pair | CPU Vectra dtype exists; Axiom CUDA bridge planned. |
| `CUDA_R_16I` / `CUDA_C_16I` | signed 16-bit integer / pair | `i16` / not exposed complex pair | CPU Vectra dtype exists; Axiom CUDA bridge planned. |
| `CUDA_R_16U` / `CUDA_C_16U` | unsigned 16-bit integer / pair | `u16` / not exposed complex pair | CPU Vectra dtype exists; Axiom CUDA bridge planned. |
| `CUDA_R_32I` / `CUDA_C_32I` | signed 32-bit integer / pair | `i32` / not exposed complex pair | CPU Vectra dtype exists; Axiom CUDA bridge planned. |
| `CUDA_R_32U` / `CUDA_C_32U` | unsigned 32-bit integer / pair | `u32` / not exposed complex pair | CPU Vectra dtype exists; Axiom CUDA bridge planned. |
| `CUDA_R_64I` / `CUDA_C_64I` | signed 64-bit integer / pair | `i64` / not exposed complex pair | CPU Vectra dtype exists; Axiom CUDA bridge planned. |
| `CUDA_R_64U` / `CUDA_C_64U` | unsigned 64-bit integer / pair | `u64` / not exposed complex pair | CPU Vectra dtype exists; Axiom CUDA bridge planned. |
| `CUDA_R_8F_E4M3` / `CUDA_R_8F_E5M2` / `CUDA_R_8F_UE8M0` | fp8 formats | Not exposed | Planned. |
| `CUDA_R_6F_E2M3` / `CUDA_R_6F_E3M2` | fp6 formats | Not exposed | Planned. |
| `CUDA_R_4F_E2M1` | fp4 format | Not exposed | Planned. |

The same data is available to code through `vx.axiom_cuda`:

- `cudaDTypeSupportRecords()`
- `findCudaDTypeSupport(cuda_name)`
- `findVectraDTypeSupport(dtype)`
- `cudaDTypeNativeSeedCount()`
- `cudaDTypeWidenedSeedCount()`
- `cudaDTypeBridgeCount()`
- `cudaDTypeSupportFingerprint()`

Current registry summary:

| Counter | Expected value | Meaning |
| --- | ---: | --- |
| `cudaDTypeSupportRecords().len` | 34 | CUDA dtype names mirrored from `cudaDataType_t` in the local CUDA headers |
| `cudaDTypeNativeSeedCount()` | 1 | `CUDA_R_32F` / `Array(f32)` native Axiom CUDA seed |
| `cudaDTypeWidenedSeedCount()` | 2 | `CUDA_R_16F` / `Array(f16)` and `CUDA_R_16BF` / `Array(BFloat16)` widened-to-f32 CUDA seeds |
| `cudaDTypeBridgeCount()` | 3 | All current Axiom CUDA-bridged Vectra dtypes |

## Current bridge behavior

- `Array(f32)` is the native CUDA seed path.
- `Array(f16)` and `Array(BFloat16)` now exercise Axiom CUDA through a widening
  seed: convert inputs to f32, run Axiom's f32 CUDA kernels where CUDA execution
  is required, then narrow outputs back to the original dtype.  Elementwise
  provenance now uses Axiom-owned widened runtime reports
  (`runTensorElementwiseBinaryF16Widened` /
  `runTensorElementwiseBinaryBF16Widened`) instead of Vectra-local report
  reconstruction.
- The widening seed is useful for API integration, smoke tests, policy routing,
  and downstream code shape, but it is not native f16/BF16 device code and does
  not claim tensor-core throughput.
- Native dtype lowering should be implemented in Axiom before production claims
  for f16/BF16/fp8/fp6/fp4/int tensor kernels.

## Validation commands

```sh
zig build -Daxiom-cuda=true -Daxiom-cuda-expect=ran axiom-cuda-smoke
zig build -Daxiom-cuda-dispatch=true axiom-cuda-dispatch-smoke
```

The CUDA smoke JSON includes `f16_add_ok`, `f16_matmul_ok`, `bf16_add_ok`, and
`bf16_matmul_ok` fields when the optional bridge is enabled. It also includes
`dtype_support_count`, `dtype_bridge_count`, `dtype_native_seed_count`,
`dtype_widened_seed_count`, and `dtype_support_fingerprint` so CI can detect
unexpected dtype-registry drift.
