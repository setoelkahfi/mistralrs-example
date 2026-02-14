# Upstream Patches for mistral.rs

This directory contains patches for the upstream [mistral.rs](https://github.com/EricLBuehler/mistral.rs) project that add tvOS Metal support.

## Patch: tvOS Metal Shader Compilation Support

**File:** `0001-feat-add-tvos-metal-shader-compilation-support.patch`

### Problem

When building mistral.rs with `--features metal` for tvOS targets (`aarch64-apple-tvos` or `aarch64-apple-tvos-sim`), the build fails with missing `KERNELS` symbols. This happens because:

1. The `Platform` enum in the Metal build scripts for `mistralrs-quant` and `mistralrs-paged-attn` only has `MacOS` and `Ios` variants — there is no `TvOS` variant, so no `.metallib` is compiled for the `appletvos` SDK.
2. The Rust source files that embed the precompiled Metal libraries via `include_bytes!` only have `#[cfg(target_os = "macos")]` and `#[cfg(target_os = "ios")]` conditions — there is no `#[cfg(target_os = "tvos")]` variant, so the `KERNELS` constant is undefined when targeting tvOS.

### What the Patch Changes

Four files are modified:

| File | Change |
|------|--------|
| `mistralrs-quant/build.rs` | Add `Platform::TvOS` → `appletvos` SDK, compile `mistralrs_quant_tvos.metallib`, write dummy in skip-precompile path |
| `mistralrs-quant/src/metal_kernels/mod.rs` | Add `#[cfg(target_os = "tvos")]` `KERNELS` constant pointing to the tvOS metallib |
| `mistralrs-paged-attn/build.rs` | Add `Platform::TvOS` → `appletvos` SDK, compile `mistralrs_paged_attention_tvos.metallib`, write dummy in skip-precompile path |
| `mistralrs-paged-attn/src/metal/kernels/mod.rs` | Add `#[cfg(target_os = "tvos")]` `KERNELS` constant pointing to the tvOS metallib |

### Metal Standard Versions

- `mistralrs-quant` uses **Metal 3.1** (`metal3.1`) for all platforms including tvOS
- `mistralrs-paged-attn` uses **Metal 3.0** (`metal3.0`) for all platforms including tvOS

Apple TV 4K (3rd generation, 2022) with A15 Bionic supports Metal GPU family Apple 8, which includes Metal 3.1 support.

### How to Apply

From the root of a mistral.rs checkout:

```sh
git am /path/to/0001-feat-add-tvos-metal-shader-compilation-support.patch
```

Or to apply without committing:

```sh
git apply /path/to/0001-feat-add-tvos-metal-shader-compilation-support.patch
```

### Known Limitations

#### candle-core `PRIVATE_RESOURCE_OPTIONS`

The upstream `candle-core` crate (v0.9.2) has a `PRIVATE_RESOURCE_OPTIONS` constant that uses `StorageModeShared` on iOS and `StorageModePrivate` on all other platforms (including tvOS). Ideally tvOS should match iOS behavior (`StorageModeShared`) since tvOS devices have unified memory. In practice, `StorageModePrivate` likely works on modern Apple TV hardware, but a separate upstream fix to candle-core would be more correct:

```rust
// candle-core/src/metal_backend/device.rs
#[cfg(any(target_os = "ios", target_os = "tvos"))]
pub const PRIVATE_RESOURCE_OPTIONS: MTLResourceOptions = MTLResourceOptions::StorageModeShared;
#[cfg(not(any(target_os = "ios", target_os = "tvos")))]
pub const PRIVATE_RESOURCE_OPTIONS: MTLResourceOptions = MTLResourceOptions::StorageModePrivate;
```

#### tvOS Simulator

The patch uses the `appletvos` SDK for device builds. For simulator targets (`aarch64-apple-tvos-sim`), xcrun may need the `appletvsimulator` SDK instead. The current approach mirrors how iOS is handled (only `iphoneos` SDK, no separate `iphonesimulator` compilation), and Metal shader libraries are typically architecture-agnostic within the same GPU family, so the device metallib should work for simulator builds as well.

#### macOS Thread QoS in mistralrs-core

`mistralrs-core/src/attention/backends/cpu.rs` uses `#[cfg(target_os = "macos")]` to set thread QoS via `pthread_set_qos_class_self_np`. This API is also available on tvOS. A separate enhancement could extend this to tvOS:

```rust
#[cfg(any(target_os = "macos", target_os = "tvos"))]
unsafe fn set_thread_affinity() { /* ... */ }
```
