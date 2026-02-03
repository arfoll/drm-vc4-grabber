# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Rust tool that captures screenshots from DRM (Direct Rendering Manager) framebuffers and streams them to a Hyperion ambient lighting server. Originally designed for Raspberry Pi VC4 graphics but also supports Intel GPUs.

## Build Commands

```bash
# Build for local development (native)
cargo build --release

# Cross-compile for Raspberry Pi (aarch64)
rustup target install aarch64-unknown-linux-gnu
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=/usr/bin/aarch64-linux-gnu-gcc
cargo build --release --target aarch64-unknown-linux-gnu

# Optimized native build (for Intel/x86)
RUSTFLAGS="-C target-cpu=native -C opt-level=3" cargo build --release
```

## Architecture

### Core Flow
1. `main.rs` - CLI entry point, opens DRM device, finds active framebuffer via CRTC/plane enumeration
2. `dump_image.rs` - Main framebuffer capture logic, handles various pixel formats and tiling modes
3. `hyperion.rs` - FlatBuffer-based TCP protocol for sending images to Hyperion server

### Pixel Format Handling (`dump_image.rs`)
The codebase handles multiple DRM pixel formats with different tiling/modifier combinations:
- **XRGB8888/ARGB8888**: Broadcom T-tiled (VC4) or linear
- **XRGB2101010/ARGB2101010**: 10-bit HDR formats, supports Intel X-tiling with fused untile+decimate
- **YUV420/NV12/P030**: Video formats with Broadcom SAND128 modifier (column-based layout)
- **RGB565**: Linear 16-bit format

### Key Concepts
- **DRM modifiers**: Describe memory layout (linear, Broadcom T-tiled, Broadcom SAND128, Intel X-tiled)
- **Decimation**: All capture paths downsample images (typically 4x-6x) to reduce bandwidth to Hyperion
- **Persistent mapping** (`PersistentMap`): Caches mmap'd framebuffers for Intel X-tiled to avoid per-frame overhead

### FFI Layer (`ffi.rs`)
Custom ioctl wrappers for DRM operations not in drm-rs:
- `fb_cmd2` (0xCE): Get framebuffer info including modifiers
- `prime_handle_to_fd`: Convert GEM handle to dma-buf FD for mmap
- `gem_close`: Release GEM handles

### Image Decoding (`image_decoder.rs`)
Contains tiled memory layout decoders for Broadcom VC4's 32x32 T-tiling pattern (alternating row directions) and YUV420→RGB conversion.
