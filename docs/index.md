## Why mistral.rs?

It runs on [Metal](https://developer.apple.com/metal/)!

## How to run?

Setup your Rust toolchain, then pick an example:

### Image Generation

Generate an image using FLUX.1-schnell (diffusion model):

```bash
# With default prompt
cargo run -- image

# With custom prompt
cargo run -- image --prompt "A cat riding a bicycle on the moon, oil painting"

# With seed prompt (auto-enhanced before generation)
cargo run -- image --seed "lonely astronaut, watercolor"

# With seed prompt and a specific enhancer model
cargo run -- image --seed "lonely astronaut" --model gemma-e2b
```

### Prompt Enhancer

Expand a short description into a detailed image-generation prompt:

```bash
# With default seed and default model (Gemma 3n E4B)
cargo run -- prompt

# With custom seed
cargo run -- prompt --seed "A lonely astronaut, watercolor"

# With a specific model
cargo run -- prompt --model gemma-e2b
cargo run -- prompt --model phi-3.5-mini --seed "cyberpunk city at night"
```

## Overview

A modular Rust example for prompt enhancement and image generation powered by [mistral.rs](https://github.com/EricLBuehler/mistral.rs). Supports macOS (Metal), Linux (CUDA), and CPU-only builds out of the box.

## Enhancer Models

The prompt enhancer supports multiple text models via the `--model` flag. Each preset is tuned with the optimal dtype / quantization strategy for its size class.

| CLI value      | Model                  | HuggingFace ID                    | Strategy | ~Memory | Best for                       |
| -------------- | ---------------------- | --------------------------------- | -------- | ------- | ------------------------------ |
| `gemma-e2b`    | Gemma 3n E2B           | `google/gemma-3n-E2B-it`          | Q4K ISQ  | ~1.5 GB | iPhone / on-device inference   |
| `gemma-e4b`    | Gemma 3n E4B (default) | `google/gemma-3n-E4B-it`          | F16      | ~8 GB   | macOS desktop (≥16 GB RAM)     |
| `phi-3.5-mini` | Phi-3.5-mini           | `microsoft/Phi-3.5-mini-instruct` | Q4K ISQ  | ~2.8 GB | Strongest quality, desktop use |

If `--model` is omitted, `gemma-e4b` is used by default.

### Choosing a model

- **On iPhone / iPad** — use `gemma-e2b`. It's quantised to Q4K (~1.5 GB) and leaves plenty of room for the diffusion model and iOS itself.
- **On a Mac with ≥16 GB RAM** — use `gemma-e4b` (the default). Full F16 gives the best quality-to-speed ratio on Apple Silicon.
- **Best prompt quality** — use `phi-3.5-mini`. Slightly larger than E2B but produces richer, more detailed prompt expansions.

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

# Enhance a seed prompt before generation (uses default model)
cargo run --release -- image --seed "lonely astronaut, watercolor"

# Enhance with a specific model
cargo run --release -- image --seed "lonely astronaut" --model gemma-e2b
```

### Prompt Enhancer

Expand a short description into a detailed image-generation prompt:

```bash
# Default seed, default model (gemma-e4b)
cargo run --release -- prompt

# Custom seed
cargo run --release -- prompt --seed "A lonely astronaut, watercolor"

# Use the lightweight on-device model
cargo run --release -- prompt --model gemma-e2b

# Use Phi-3.5-mini for highest quality
cargo run --release -- prompt --model phi-3.5-mini --seed "cyberpunk city at night"
```

### Help

```bash
cargo run -- --help
cargo run -- image --help
cargo run -- prompt --help
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
4. **Recommended model** — Use `gemma-e2b` for on-device inference. It's quantised to Q4K (~1.5 GB) and designed by Google specifically for edge / mobile deployment.

Build for iOS:

```bash
rustup target add aarch64-apple-ios
cargo build --release --target aarch64-apple-ios
```

## Project Structure

```
mistralrs-example/
├── src/
│   ├── main.rs               # CLI entry point (clap subcommands + --model flag)
│   ├── image_generation.rs    # FLUX.1-schnell diffusion image generation
│   └── promp_enhancer.rs      # Prompt enhancement (EnhancerModel presets + PromptEnhancer)
├── asm/
│   └── chkstk_darwin_stub.s   # iOS linker stub for aws-lc-sys
├── .cargo/
│   └── config.toml            # iOS target features (FP16)
├── .github/
│   └── workflows/
│       └── build-check.yml    # CI: matrix build check (macOS + iOS cross-compile)
├── build.rs                   # Conditional iOS stub compilation
├── Cargo.toml                 # Multi-platform dependency configuration
└── README.md
```

## License

MIT.
