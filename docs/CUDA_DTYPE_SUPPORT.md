# CUDA dtype support matrix

This matrix records CUDA dtype names visible in the local CUDA headers and how
Vectra maps supported entries through the default Axiom accelerator backend.  It is
an integration status document, not a claim that native CUDA kernels exist for
all listed dtypes.

Local CUDA evidence: `/usr/local/cuda/include/library_types.h` declares
`cudaDataType_t` values including:

| CUDA dtype | Meaning in CUDA headers | Vectra dtype | Axiom bridge status |
| --- | --- | --- | --- |
| `CUDA_R_16F` | real half | `f16` | Same-shape add/sub/mul/div now try Axiom's native f16 CUDA runtime seed first, with widened-to-f32 fallback. Contiguous 2D matmul calls Axiom's typed SIMT GEMM runtime seed, which reports typed launch-plan readiness and the explicit `widened_f32_cuda_compute` route while using widened f32 compute underneath today. |
| `CUDA_C_16F` | complex half pair | Not exposed | Planned. |
| `CUDA_R_16BF` | real bfloat16 | `BFloat16` | Same-shape add/sub/mul/div now try Axiom's native BF16 CUDA runtime seed first, with widened-to-f32 fallback. Contiguous 2D matmul calls Axiom's typed SIMT GEMM runtime seed, which reports typed launch-plan readiness and the explicit `widened_f32_cuda_compute` route while using widened f32 compute underneath today; CUTILE tensor-core lowering remains future work. |
| `CUDA_C_16BF` | complex bfloat16 pair | Not exposed | Planned. |
| `CUDA_R_32F` | real float | `f32` | Native Axiom CUDA seed for add/sub/mul/div and SAXPY; owning CUDA Array matmul/matmulAdd use Axiom cached cuBLAS SGEMM for production throughput with the CUDA Tile IR seed retained as fallback/provenance. |
| `CUDA_C_32F` | complex float pair | `Complex64` | CPU Vectra support exists; Axiom CUDA bridge planned. |
| `CUDA_R_64F` | real double | `f64` | Axiom CPU→Veyra path for supported ops plus native Axiom CUDA device same-shape elementwise and DGEMM matmul seeds. |
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
| `cudaDTypeNativeSeedCount()` | 2 | `CUDA_R_32F` / `Array(f32)` and `CUDA_R_64F` / `Array(f64)` native Axiom CUDA seeds |
| `cudaDTypeWidenedSeedCount()` | 2 | `CUDA_R_16F` / `Array(f16)` and `CUDA_R_16BF` / `Array(BFloat16)` widened-to-f32 CUDA seeds |
| `cudaDTypeBridgeCount()` | 4 | All current Axiom CUDA-bridged Vectra dtypes |

## Current bridge behavior

- `Array(f32)` and `Array(f64)` are native CUDA seed paths; f64 currently covers same-shape/scalar elementwise, maximum/addcmul/addcdiv/lerp/neg/abs/reciprocal/square/powScalar(0/1/2/3)/sqrt/rsqrt/exp/relu/threshold/leakyRelu/relu6/clip/clipArray/elu/celu/sigmoid/silu/hardsigmoid/hardswish/softsign/softshrink, DGEMM matmul, and matmulAdd/fusion.
- `Array(f16)` and `Array(BFloat16)` now try Axiom's native typed CUDA
  elementwise runtime seeds for same-shape add/sub/mul/div before falling back to
  widened f32 routes; f16 and BFloat16 widened activation/powScalar combinations such as
  relu/sigmoid/softsign/clip/powScalar(0/1/2/3) are covered by the CUDA device smoke.
- `Array(f16)` and `Array(BFloat16)` matmul now exercise Axiom CUDA through
  Axiom's typed SIMT GEMM runtime seed entry points.  Those entry points report
  typed launch-plan/readiness/seed fingerprints and the explicit
  `widened_f32_cuda_compute` route while still using widened f32 GEMM compute
  underneath today, then narrow outputs back to the original dtype.
  Elementwise provenance now uses Axiom-owned widened runtime reports
  (`runTensorElementwiseBinaryF16Widened` /
  `runTensorElementwiseBinaryBF16Widened`) instead of Vectra-local report
  reconstruction; those reports can include f32 CUDA compute-run fingerprints
  when Axiom delegates the widened compute step to its f32 CUDA runtime.
- The native typed elementwise seeds are useful for API integration, smoke tests,
  and policy routing, but they are not tensor-core/CUTILE throughput paths.
- Tensor-core/CUTILE-native dtype lowering should be implemented in Axiom before
  production throughput claims for f16/BF16/fp8/fp6/fp4/int tensor kernels.

## Validation commands

```sh
zig build -Daxiom-cuda-expect=ran axiom-cuda-smoke
zig build axiom-cuda-dispatch-smoke
```

The CUDA smoke JSON includes `f16_add_ok`, `f16_matmul_ok`, `bf16_add_ok`,
`bf16_matmul_ok`, `f64_matmul_ok`, `f64_elementwise_ok`, and `f64_matmul_add_ok` fields when the CUDA smokes run. It also includes
`f16_native_execution_fingerprint` and `bf16_native_execution_fingerprint` when
the native typed elementwise seeds run,
`dtype_support_count`, `dtype_bridge_count`, `dtype_native_seed_count`,
`dtype_widened_seed_count`, and `dtype_support_fingerprint` so CI can detect
unexpected dtype-registry drift.
