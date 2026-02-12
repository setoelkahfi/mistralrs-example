## Overview

A modular Rust example for prompt enhancement and image generation powered by [mistral.rs](https://github.com/EricLBuehler/mistral.rs). Supports macOS (Metal), Linux (CUDA), and CPU-only builds out of the box.

## How to run?

## Multi-Platform GPU Support

The project automatically selects the right GPU backend based on your target platform — no manual feature flags needed for the default case.

| Platform        | GPU Backend                  | Activation                                            |
| --------------- | ---------------------------- | ----------------------------------------------------- |
| **macOS / iOS** | Metal (Apple GPU)            | Automatic — enabled via target-conditional dependency |
| **Linux**       | CUDA (NVIDIA GPU)            | `cargo build --features cuda`                         |
| **Linux**       | CUDA + FlashAttention        | `cargo build --features flash-attn`                   |
| **Linux**       | CUDA + cuDNN                 | `cargo build --features cudnn`                        |
| **macOS**       | Accelerate (CPU, Apple BLAS) | `cargo build --features accelerate`                   |
| **Linux**       | MKL (Intel CPU)              | `cargo build --features mkl`                          |
| **Any**         | CPU only                     | `cargo build` (no features)                           |

### How It Works

The `metal` feature is gated behind `cfg(target_os = "macos")` and `cfg(target_os = "ios")` in `Cargo.toml`, so Apple-only crates like `objc2` are never compiled on Linux or Windows. On non-Apple platforms the build defaults to CPU unless you explicitly enable CUDA or another backend.

## Prerequisites

1. **Rust toolchain** — install via [rustup](https://rustup.rs/)
2. **macOS**: Xcode Command Line Tools (`xcode-select --install`)
3. **Linux + CUDA**: NVIDIA driver, CUDA toolkit, and optionally cuDNN
4. **iOS cross-compilation** (optional): see [iOS Build Notes](#ios-build-notes) below

## How to Run

### Image Generation

Generate an image using FLUX.1-schnell (diffusion model):

```bash
# Default prompt
cargo run --release -- image

# Custom prompt
cargo run --release -- image --prompt "A cat riding a bicycle on the moon, oil painting"
```

### Prompt Enhancer

Expand a short description into a detailed image-generation prompt using Phi-3.5-mini:

```bash
# Default seed
cargo run --release -- prompt

# Custom seed
cargo run --release -- prompt --seed "A lonely astronaut, watercolor"
```

### Help

```bash
cargo run -- --help
```

## Available Features

| Feature      | Description                           |
| ------------ | ------------------------------------- |
| `cuda`       | NVIDIA CUDA support                   |
| `flash-attn` | FlashAttention (implies `cuda`)       |
| `cudnn`      | cuDNN acceleration (requires CUDA)    |
| `accelerate` | Apple Accelerate framework (CPU BLAS) |
| `mkl`        | Intel MKL (CPU BLAS)                  |

Metal is **not** a feature flag — it is enabled automatically on Apple platforms.

## iOS Build Notes

For cross-compiling to iOS (e.g. for a Tauri mobile app):

1. **FP16 NEON instructions** — `.cargo/config.toml` already enables `+fp16` for `aarch64-apple-ios` and `aarch64-apple-ios-sim` targets.
2. **`___chkstk_darwin` stub** — `aws-lc-sys` (pulled in by `rustls`) references a macOS-only symbol. The included `asm/chkstk_darwin_stub.s` provides a no-op stub that the build script compiles automatically for iOS targets.
3. **Production TLS** — For real iOS apps, consider using a native TLS backend instead of `rustls` + `aws-lc-sys`.

Build for iOS:

```bash
rustup target add aarch64-apple-ios
cargo build --release --target aarch64-apple-ios
```

## Project Structure

```
mistralrs-example/
├── src/
│   ├── main.rs               # CLI entry point (clap subcommands)
│   ├── image_generation.rs    # FLUX.1-schnell diffusion image generation
│   └── promp_enhancer.rs      # Phi-3.5-mini prompt enhancement
├── asm/
│   └── chkstk_darwin_stub.s   # iOS linker stub for aws-lc-sys
├── .cargo/
│   └── config.toml            # iOS target features (FP16)
├── build.rs                   # Conditional iOS stub compilation
├── Cargo.toml                 # Multi-platform dependency configuration
└── README.md
```

## License

MIT.
