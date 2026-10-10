use std::{convert::TryFrom, mem::size_of, os::fd::AsRawFd};
use std::sync::atomic::{AtomicBool, Ordering};

use drm::control::framebuffer::Handle;
use drm::SystemError;
use drm_fourcc::{DrmFourcc, DrmModifier};
use image::{GenericImage, RgbImage};
use libc::close;
use nix::sys::mman;

use crate::{
    ffi::{self, gem_close},
    image_decoder::{
        decode_image, decode_image_multichannel, decode_small_image_multichannel,
        decode_tiled_small_image, rgb565_to_rgb888, ToRgb, YUV420Pixel,
    },
    Card,
};

use std::os::fd::{IntoRawFd, RawFd};
use nix::sys::mman::{ProtFlags, MapFlags};
use once_cell::sync::OnceCell;
use std::collections::HashMap;
use std::sync::Mutex;

/// Global flag for proper PQ (ST.2084) HDR tone mapping
static HDR_PQ_MODE: AtomicBool = AtomicBool::new(false);

/// Luminance multipliers for HDR and SDR modes (stored as f32 bits)
static LUMINANCE_HDR: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0x3f800000); // 1.0
static LUMINANCE_SDR: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0x3f800000); // 1.0
/// Saturation boost for HDR content (stored as f32 bits)
static SATURATION_HDR: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0x3f800000); // 1.0

/// Enable proper PQ HDR tone mapping
pub fn set_hdr_pq_mode(enabled: bool) {
    HDR_PQ_MODE.store(enabled, Ordering::Relaxed);
}

/// Set luminance multiplier for HDR content (default 1.0)
pub fn set_luminance_hdr(value: f32) {
    LUMINANCE_HDR.store(value.to_bits(), Ordering::Relaxed);
}

/// Set luminance multiplier for SDR content (default 1.0)
pub fn set_luminance_sdr(value: f32) {
    LUMINANCE_SDR.store(value.to_bits(), Ordering::Relaxed);
}

/// Set saturation boost for HDR content (default 1.0, try 1.2-1.5)
pub fn set_saturation_hdr(value: f32) {
    SATURATION_HDR.store(value.to_bits(), Ordering::Relaxed);
}

// SAFETY: PersistentMap contains only an FD and a read-only mmap pointer.
// We never mutate the mapped memory, and DRM guarantees it remains valid
// while the handle is alive, so it is safe to share references between threads.
unsafe impl Send for PersistentMap {}
unsafe impl Sync for PersistentMap {}

/// Global cache of persistent DRM mappings per framebuffer handle
static PERSISTENT_MAPS: OnceCell<Mutex<HashMap<u32, PersistentMap>>> = OnceCell::new();

fn get_persistent_map(
    card: &Card,
    handle: u32,
    length_words: usize,
    verbose: bool,
) -> Result<&'static PersistentMap, SystemError> {
    let maps = PERSISTENT_MAPS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut lock = maps.lock().unwrap();

    if !lock.contains_key(&handle) {
        let map = PersistentMap::new(card, handle, length_words, verbose)?;
        lock.insert(handle, map);
    }

    // SAFETY: OnceCell + Mutex ensures stable address
    let map_ref: *const PersistentMap = lock.get(&handle).unwrap();
    unsafe { Ok(&*map_ref) }
}


/// Persistent memory map for a DRM framebuffer (u32 pixels)
pub struct PersistentMap {
    pub fd: RawFd,
    pub ptr: *const u32,
    pub length_words: usize,
}

impl PersistentMap {
    pub fn new(card: &Card, handle: u32, length_words: usize, verbose: bool) -> Result<Self, SystemError> {
        let fd = ffi::prime_handle_to_fd(card.as_raw_fd(), handle)?;
        let length_bytes = length_words * std::mem::size_of::<u32>();

        if verbose {
            println!("PersistentMap: mapping handle {} ({} words)", handle, length_words);
        }

        let addr = std::ptr::null_mut();
        let prot = ProtFlags::PROT_READ;
        let flags = MapFlags::MAP_SHARED;
        let map = unsafe {
            mman::mmap(addr, length_bytes, prot, flags, fd, 0)
                .map_err(|_| SystemError::Unknown { errno: nix::errno::Errno::ENOMEM })?
        } as *const u32;

        Ok(Self {
            fd,
            ptr: map,
            length_words,
        })
    }

    #[inline]
    pub fn as_slice(&self) -> &[u32] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.length_words) }
    }
}

impl Drop for PersistentMap {
    fn drop(&mut self) {
        let length_bytes = self.length_words * std::mem::size_of::<u32>();
        unsafe {
            let _ = mman::munmap(self.ptr as *mut _, length_bytes);
            libc::close(self.fd);
        }
    }
}

const NUM_SAMPLES: usize = 32;

/// Quick-sample a framebuffer: mmap, read NUM_SAMPLES evenly-spaced u32 values, munmap.
/// Used to detect unchanged frames without doing the full decode.
pub fn sample_framebuffer(
    card: &Card,
    fb: Handle,
) -> Result<[u32; NUM_SAMPLES], SystemError> {
    let fbinfo2 = ffi::fb_cmd2(card.as_raw_fd(), fb.into())?;
    let handle = fbinfo2.handles[0];
    let pitch = fbinfo2.pitches[0] as usize;
    let height = fbinfo2.height as usize;
    let total_bytes = pitch * height;
    let total_words = total_bytes / 4;

    let hfd = ffi::prime_handle_to_fd(card.as_raw_fd(), handle)?;
    let mut samples = [0u32; NUM_SAMPLES];

    unsafe {
        let map = mman::mmap(
            core::ptr::null_mut(),
            total_bytes,
            mman::ProtFlags::PROT_READ,
            mman::MapFlags::MAP_SHARED,
            hfd,
            0,
        )
        .map_err(|_| SystemError::Unknown {
            errno: nix::errno::Errno::ENOMEM,
        })?;

        let slice = std::slice::from_raw_parts(map as *const u32, total_words);
        let step = total_words / NUM_SAMPLES;
        for i in 0..NUM_SAMPLES {
            samples[i] = slice[i * step];
        }

        let _ = mman::munmap(map, total_bytes);
        close(hfd);
    }

    let _ = gem_close(card.as_raw_fd(), handle);
    Ok(samples)
}

fn copy_buffer<T: Sized + Copy>(
    card: &Card,
    handle: u32,
    to: &mut [T],
    verbose: bool,
) -> Result<(), SystemError> {
    let length = to.len() * size_of::<T>();

    let hfd = ffi::prime_handle_to_fd(card.as_raw_fd(), handle)?;

    if verbose {
        println!("handle fd {}", hfd);
    }

    let addr = core::ptr::null_mut();
    let prot = mman::ProtFlags::PROT_READ;
    let flags = mman::MapFlags::MAP_SHARED;
    unsafe {
        let map = mman::mmap(addr, length as _, prot, flags, hfd, 0).unwrap();

        let mapping: &mut [T] = std::slice::from_raw_parts_mut(map as *mut _, to.len());
        to.copy_from_slice(mapping);
        mman::munmap(map, length as _).unwrap();

        if close(hfd) == -1 {
            panic!("Failed to close prime fd.");
        };
    }

    Ok(())
}

fn decimate_image_4(size: (usize, usize), image: &[u32], copy: &mut [u32], border_frac: f32) {
    let decim = (4, 4);
    let newsize = (size.0 / decim.0, size.1 / decim.1);

    let border_active = border_frac > 0.0;
    let bt = (size.1 as f32 * border_frac) as usize;
    let bl = (size.0 as f32 * border_frac) as usize;
    let bb = size.1.saturating_sub(bt);
    let br = size.0.saturating_sub(bl);

    for y in 0..newsize.1 {
        let ty = decim.1 * y;
        let y_interior = border_active && ty >= bt && ty < bb;
        for x in 0..newsize.0 {
            if y_interior {
                let tx = decim.0 * x;
                if tx >= bl && tx < br {
                    continue;
                }
            }
            let tx = decim.0 * x;
            copy[y * newsize.0 + x] = image[ty * size.0 + tx];
        }
    }
}

/// ST.2084 PQ EOTF: decode PQ-encoded value to linear light (0-1 normalized)
#[inline]
fn pq_eotf(v: f32) -> f32 {
    const M1: f32 = 0.1593017578125;
    const M2: f32 = 78.84375;
    const C1: f32 = 0.8359375;
    const C2: f32 = 18.8515625;
    const C3: f32 = 18.6875;

    let vp = v.powf(1.0 / M2);
    let num = (vp - C1).max(0.0);
    let den = C2 - C3 * vp;
    (num / den).powf(1.0 / M1)
}

/// Lookup tables for the per-channel part of `ten_to_eight`.
///
/// The 10-bit code space has only 1024 values, and the nonlinear chain
/// (PQ EOTF -> tone map -> gamma, or the simple SDR power curve) depends only
/// on those fixed constants, not on the runtime luminance/saturation. So the
/// expensive `powf` chain is evaluated 1024 times up front instead of once per
/// output pixel per frame. This removes the dominant cost of 10-bit capture.
static PQ_CHANNEL_LUT: OnceCell<[f32; 1024]> = OnceCell::new();
static SDR_CHANNEL_LUT: OnceCell<[f32; 1024]> = OnceCell::new();

#[inline]
fn pq_channel_lut() -> &'static [f32; 1024] {
    PQ_CHANNEL_LUT.get_or_init(|| {
        const EXPOSURE: f32 = 10.0;
        let mut lut = [0f32; 1024];
        for (i, slot) in lut.iter_mut().enumerate() {
            let v = i as f32 / 1023.0;
            let lin = pq_eotf(v);
            // Reinhard tone map + sRGB gamma encode
            let tonemapped = (lin * EXPOSURE) / (1.0 + lin * EXPOSURE);
            *slot = tonemapped.powf(1.0 / 2.2);
        }
        lut
    })
}

#[inline]
fn sdr_channel_lut() -> &'static [f32; 1024] {
    SDR_CHANNEL_LUT.get_or_init(|| {
        const GAMMA: f32 = 0.55;
        let mut lut = [0f32; 1024];
        for (i, slot) in lut.iter_mut().enumerate() {
            *slot = (i as f32 / 1023.0).powf(GAMMA);
        }
        lut
    })
}

/// Convert in-place from XRGB2101010/ARGB2101010 to XRGB8888 order.
/// Input: 0b XX RRRRRRRRRR GGGGGGGGGG BBBBBBBBBB (bits: 31..0)
/// Output (LE memory): u32 with bytes [BB, GG, RR, 00]
///
/// Two modes:
/// - Default: Simple power curve to counteract PQ encoding
/// - HDR_PQ_MODE: Proper PQ decode + Reinhard tone mapping
#[inline]
fn ten_to_eight(v10: u32) -> (u32, u32, u32) {
    // Extract full 10-bit values; they are LUT indices, not normalized inputs.
    let r10 = ((v10 >> 20) & 0x3FF) as usize;
    let g10 = ((v10 >> 10) & 0x3FF) as usize;
    let b10 = (v10 & 0x3FF) as usize;

    if HDR_PQ_MODE.load(Ordering::Relaxed) {
        let lut = pq_channel_lut();
        let lum = f32::from_bits(LUMINANCE_HDR.load(Ordering::Relaxed));
        let sat = f32::from_bits(SATURATION_HDR.load(Ordering::Relaxed));

        // PQ decode + tone map + gamma all come from the LUT; only the
        // cross-channel saturation and luminance scaling remain per pixel.
        let mut r = lut[r10];
        let mut g = lut[g10];
        let mut b = lut[b10];

        // Boost saturation to compensate for BT.2020 -> BT.709 gamut loss
        let avg = (r + g + b) / 3.0;
        r = avg + (r - avg) * sat;
        g = avg + (g - avg) * sat;
        b = avg + (b - avg) * sat;

        // Apply luminance and convert to 8-bit
        let scale = lum * 255.0;
        let r8 = (r * scale).clamp(0.0, 255.0) as u32;
        let g8 = (g * scale).clamp(0.0, 255.0) as u32;
        let b8 = (b * scale).clamp(0.0, 255.0) as u32;

        (r8, g8, b8)
    } else {
        let lut = sdr_channel_lut();
        let lum = f32::from_bits(LUMINANCE_SDR.load(Ordering::Relaxed));

        let scale = lum * 255.0;
        let r8 = (lut[r10] * scale).clamp(0.0, 255.0) as u32;
        let g8 = (lut[g10] * scale).clamp(0.0, 255.0) as u32;
        let b8 = (lut[b10] * scale).clamp(0.0, 255.0) as u32;

        (r8, g8, b8)
    }
}

fn xr30_to_xr24_inplace(pixels: &mut [u32]) {
    for p in pixels.iter_mut() {
        let (r8, g8, b8) = ten_to_eight(*p);
        // Numeric u32 is 0x00RRGGBB (conventional). In little-endian memory
        // this becomes bytes [BB, GG, RR, 00], which matches XR24 expectations.
        *p = (r8 << 16) | (g8 << 8) | (b8 << 0);
    }
}

fn ar30_to_ar24_inplace(pixels: &mut [u32]) {
    // Alpha is ignored for screenshots; drop to XRGB8888 layout.
    xr30_to_xr24_inplace(pixels);
}

fn decode_p030_image(
    card: &Card,
    size: (usize, usize),
    pitches: u32,
    handle: u32,
    modifier: u64,
    offset: usize,
    verbose: bool,
) -> Result<RgbImage, SystemError> {
    // We assume the DRM BROADCOM SAND128 format
    if u64::from(drm_fourcc::DrmModifier::Broadcom_sand128) != modifier & !(0xFFFF << 8) {
        panic!("Unsupported P030 modifier value");
    }

    let stride = 128 / 4; // each column is 128 bytes wide, we use 4 bytes per word
    let colpx = 96;

    let ypitch = pitches as usize / (32 / 8);
    let ylines = ((modifier >> 8) & 0xFFFFFFFF) as usize;
    let length = ylines * (size.0 / colpx) * stride;
    let crcboffset = offset / 4; // offset of the CrCb information in each column

    if verbose {
        println!(
            "P030, size: {:?}, lines: {}, pitches: {}, length: {}",
            size, ylines, ypitch, length
        );
    }

    let mut yplane = vec![0u32; length as _];
    copy_buffer(card, handle, &mut yplane, verbose)?;

    let decim = 3;
    let mut img = RgbImage::new((size.0 / decim) as _, (size.1 / decim) as _);
    for y in 0..size.1 / decim {
        let ty = y * decim;
        for x in 0..size.0 / decim {
            let tx = x * decim;
            let col = tx / colpx;
            let col_offset = col * stride * ylines;
            let x_mod = (tx % colpx) / decim;

            let ypx = unsafe { yplane.get_unchecked(col_offset + ty * stride + x_mod) };
            let rx = x_mod / 2 * 2;
            let crcind = col_offset + crcboffset + ty / 2 * stride + rx;
            let crcbpx = unsafe { yplane.get_unchecked(crcind + 1) };

            let yuv = YUV420Pixel::new((ypx >> 2) as u8, (crcbpx >> 12) as u8, (crcbpx >> 2) as u8);

            unsafe {
                img.unsafe_put_pixel(x as _, y as _, yuv.rgb());
            }
        }
    }

    Ok(img)
}

fn decode_nv12_image(
    card: &Card,
    size: (usize, usize),
    pitches: u32,
    handle: u32,
    modifier: u64,
    offset: usize,
    verbose: bool,
) -> Result<RgbImage, SystemError> {
    // We assume the DRM BROADCOM SAND128 format
    if u64::from(drm_fourcc::DrmModifier::Broadcom_sand128) != modifier & !(0xFFFF << 8) {
        panic!("Unsupported NV12 modifier value");
    }

    let stride = 128 / 4; // each column is 128 bytes wide, we use 4 bytes per word
    let colpx = 128; // 1 byte per pixel

    let ypitch = pitches as usize / (32 / 8);
    let ylines = ((modifier >> 8) & 0xFFFFFFFF) as usize;
    let length = ylines * (size.0 / colpx) * stride;
    let crcboffset = offset / 4; // offset of the CrCb information in each column

    if verbose {
        println!(
            "NV12, size: {:?}, lines: {}, pitches: {}, length: {}",
            size, ylines, ypitch, length
        );
    }

    let mut yplane = vec![0u32; length as _];
    copy_buffer(card, handle, &mut yplane, verbose)?;

    let decim: usize = 4;
    let mut img = RgbImage::new((size.0 / decim) as _, (size.1 / decim) as _);
    for y in 0..size.1 / decim {
        let ty = y * decim;
        for x in 0..size.0 / decim {
            let tx = x * decim;
            let col = tx / colpx;
            let col_offset = col * stride * ylines;
            let x_mod = (tx % colpx) / decim;

            let ypx = unsafe { yplane.get_unchecked(col_offset + ty * stride + x_mod) };
            let rx = x_mod / 2 * 2;
            let crcind = col_offset + crcboffset + ty / 2 * stride + rx;
            let crcbpx = unsafe { yplane.get_unchecked(crcind + 1) };

            let yuv = YUV420Pixel::new((ypx >> 0) as u8, (crcbpx >> 0) as u8, (crcbpx >> 8) as u8);

            unsafe {
                img.unsafe_put_pixel(x as _, y as _, yuv.rgb());
            }
        }
    }

    Ok(img)
}

fn dump_linear_to_image(
    card: &Card,
    pitch: u32,
    size: (u32, u32),
    bpp: u32,
    handle: u32,
    border_frac: f32,
    verbose: bool,
) -> Result<RgbImage, SystemError> {
    let size = (size.0, size.1);

    let length = pitch * size.1 / (bpp / 8);

    if verbose {
        println!(
            "linear, size: {:?}, pitch: {}, bpp: {}, length: {}",
            size, pitch, bpp, length
        );
    }
    let mut copy = vec![0u32; length as _];
    copy_buffer(card, handle, &mut copy, verbose)?;

    let mut dec = vec![0u32; (length / (4 * 4)) as _];
    decimate_image_4(
        (size.0 as _, size.1 as _),
        copy.as_slice(),
        dec.as_mut_slice(),
        border_frac,
    );

    Ok(decode_image(
        dec.as_mut_slice(),
        pitch / 4,
        (size.0 / 4, size.1 / 4),
    ))
}

fn dump_linear_xr30_to_image(
    card: &Card,
    pitch: u32,
    size: (u32, u32),
    _bpp: u32,
    handle: u32,
    border_frac: f32,
    verbose: bool,
) -> Result<RgbImage, SystemError> {
    // Even though it's 30bpp, DRM buffers are 32-bit words per pixel.
    let length_words = (pitch * size.1) / 4;
    if verbose {
        println!(
            "linear XR30, size: {:?}, pitch: {}, words: {}",
            size, pitch, length_words
        );
    }

    let mut copy = vec![0u32; length_words as _];
    copy_buffer(card, handle, &mut copy, verbose)?;

    // Fused XR30→RGB + decimate + border skip in a single pass
    let width = size.0 as usize;
    let height = size.1 as usize;
    let pitch_words = pitch as usize / 4;
    let decim = 4usize;
    let dw = width / decim;
    let dh = height / decim;

    let border_active = border_frac > 0.0;
    let border_top = (height as f32 * border_frac) as usize;
    let border_bot = height.saturating_sub(border_top);
    let border_left = (width as f32 * border_frac) as usize;
    let border_right = width.saturating_sub(border_left);

    let mut img = RgbImage::new(dw as u32, dh as u32);
    let dst = img.as_mut();

    for dy in 0..dh {
        let sy = dy * decim;
        let y_interior = border_active && sy >= border_top && sy < border_bot;
        for dx in 0..dw {
            let sx = dx * decim;
            if y_interior && sx >= border_left && sx < border_right {
                continue;
            }
            let px = copy[sy * pitch_words + sx];
            let (r8, g8, b8) = ten_to_eight(px);
            let di = (dy * dw + dx) * 3;
            dst[di] = r8 as u8;
            dst[di + 1] = g8 as u8;
            dst[di + 2] = b8 as u8;
        }
    }

    Ok(img)
}

fn dump_rgb565_to_image(
    card: &Card,
    pitch: u32,
    size: (u32, u32),
    bpp: u32,
    handle: u32,
    verbose: bool,
) -> Result<RgbImage, SystemError> {
    // let size = (size.0, size.1 / 64);

    let length = pitch * size.1 / (bpp / 8);

    if verbose {
        println!(
            "rgb565, size: {:?}, pitch: {}, bpp: {}, length: {}",
            size, pitch, bpp, length
        );
    }
    let mut copy = vec![0u16; length as _];
    copy_buffer(card, handle, &mut copy, verbose)?;

    Ok(rgb565_to_rgb888(copy.as_mut_slice(), pitch, size))
}

fn dump_broadcom_tiled_to_image(
    card: &Card,
    size: (u32, u32),
    bpp: u32,
    handle: u32,
    verbose: bool,
) -> Result<RgbImage, SystemError> {
    let tilesize = 32;
    let tile_count = |n| (n + tilesize - 1) / tilesize;
    let tiles = (tile_count(size.0), tile_count(size.1));
    let total_tiles = tiles.0 * tiles.1;

    let length = total_tiles * tilesize * tilesize * (bpp / 8);

    let mut copy = vec![0; (length / 4) as _];
    copy_buffer(card, handle, &mut copy, verbose)?;

    Ok(decode_tiled_small_image(
        copy.as_mut_slice(),
        tilesize,
        tiles,
        size,
    ))
}

fn dump_yuv420_to_image(
    card: &Card,
    size: (u32, u32),
    pitches: [u32; 4],
    handles: [u32; 4],
    offsets: [u32; 4],
    verbose: bool,
) -> Result<RgbImage, SystemError> {
    // The length of the entire buffer is the length of the last buffer plus its
    // offset (assuming they are in order). The U and V buffers are grouped into
    // 2x2 tiles, hence the length is divided by 4.
    let length = offsets[2] + size.1 * pitches[2] / (pitches[0] / pitches[2]);
    //println!("  -> Mounting @{} +{}", offset, length);

    let mut copy = vec![0; length as _];
    copy_buffer(card, handles[0], &mut copy, verbose)?;

    let buffer_range = |i| {
        offsets[i] as usize..(offsets[i] + size.1 * pitches[i] / (pitches[0] / pitches[i])) as usize
    };

    let mappings = [
        &copy[buffer_range(0)],
        &copy[buffer_range(1)],
        &copy[buffer_range(2)],
    ];

    let mut pitches1 = [0; 3];
    pitches1.copy_from_slice(&pitches[0..3]);

    if size.0 > 640 {
        // If the image is large then just decode a smaller image
        Ok(decode_small_image_multichannel(mappings, size, pitches1))
    } else {
        Ok(decode_image_multichannel(mappings, size, pitches1))
    }
}

/// Pick the largest integer decimation factor `<= target` that evenly divides
/// both dimensions. Guarantees the fused tile walker's `dst` indices stay in
/// bounds for resolutions that aren't cleanly divisible by `target`.
fn pick_decim(width: u32, height: u32, target: usize) -> usize {
    let mut d = target;
    while d > 1 && ((width as usize) % d != 0 || (height as usize) % d != 0) {
        d -= 1;
    }
    d
}

/// Fused untile + decimate (nearest) + pixel conversion for Intel X-tiled 32bpp.
/// - Input: any 32bpp format (XR30/AR30 10-bit or XR24/AR24 8-bit) using the
///   Intel X-tiling modifier ((1<<56)|1)
/// - `convert` maps one raw 32-bit word to an (R, G, B) triple.
/// - Output: RgbImage downscaled by integer `decim` (e.g. 6 → 3840x2160 → 640x360)
fn dump_intel_xtiled_decimated_to_image<F>(
    card: &Card,
    pitch: u32,
    size: (u32, u32),
    handle: u32,
    decim: usize,
    border_frac: f32,
    verbose: bool,
    convert: F,
) -> Result<image::RgbImage, SystemError>
where
    F: Fn(u32) -> (u8, u8, u8),
{
    // Intel X-tiled 32bpp: 128×8 pixels per tile = 4096 bytes (1024 u32 words)
    const TILE_W: usize = 128;
    const TILE_H: usize = 8;
    const TILE_WORDS: usize = TILE_W * TILE_H; // 1024 u32 per tile row-block

    let width  = size.0 as usize;
    let height = size.1 as usize;

    assert!(decim >= 2, "decim must be >=2");
    assert!(width % decim == 0 && height % decim == 0, "decim must divide dimensions");

    let dst_w = width / decim;
    let dst_h = height / decim;

    let pitch_words = pitch as usize / 4;          // 32bpp → words per scanline
    let tiles_x = (width  + TILE_W - 1) / TILE_W;  // full tiles across
    let tiles_y = (height + TILE_H - 1) / TILE_H;  // full tiles down
    let tiles_per_pitch = pitch as usize / (TILE_W * 4); // tiles per scanline in memory

    // Persistent map for the whole framebuffer (u32 view)
    let length_words = pitch_words * height;
    let map = get_persistent_map(card, handle, length_words, verbose)?;
    let src: &[u32] = map.as_slice();

    if verbose {
        println!(
            "intel x-tiled (fused decimate), src={:?}, pitch_words={}, tiles=({}x{}), decim={}",
            size, pitch_words, tiles_x, tiles_y, decim
        );
    }

    // Destination image (RGB888)
    let mut img = image::RgbImage::new(dst_w as u32, dst_h as u32);
    let dst_buf = img.as_mut(); // &mut [u8]

    // Border bounds in source pixel coordinates
    let border_active = border_frac > 0.0;
    let border_top = (height as f32 * border_frac) as usize;
    let border_bot = height.saturating_sub(border_top);
    let border_left = (width as f32 * border_frac) as usize;
    let border_right = width.saturating_sub(border_left);

    // Walk tiles; only touch rows/cols that land on decim grid
    for ty in 0..tiles_y {
        let tile_y0 = ty * TILE_H;
        // For each row inside the tile
        for y in 0..TILE_H {
            let src_y = tile_y0 + y;
            if src_y >= height { break; }

            // Only keep every `decim`-th row
            if src_y % decim != 0 { continue; }
            let dst_y = src_y / decim;

            let y_in_interior = border_active && src_y >= border_top && src_y < border_bot;

            for tx in 0..tiles_x {
                let tile_x0 = tx * TILE_W;

                // Tile-level skip: entire tile is in the interior
                if y_in_interior {
                    let tile_x_end = (tile_x0 + TILE_W).min(width);
                    if tile_x0 >= border_left && tile_x_end <= border_right {
                        continue;
                    }
                }

                // Base index of this tile in the linear (tiled) buffer
                // Each tile row-block is TILE_WORDS words; tiles laid out left→right along pitch
                let tile_base = (ty * tiles_per_pitch + tx) * TILE_WORDS;
                let src_row_start = tile_base + y * TILE_W;

                // For each column inside the tile, step by decim
                // Clamp the last tile to the actual width
                let run = TILE_W.min(width.saturating_sub(tile_x0));
                let mut x = 0usize;
                while x < run {
                    let src_x = tile_x0 + x;
                    if src_x % decim == 0 {
                        // Pixel-level border skip for tiles straddling the border
                        if !(y_in_interior && src_x >= border_left && src_x < border_right) {
                            let dst_x = src_x / decim;

                            // Fetch pixel (u32) and convert to RGB
                            let px = unsafe { *src.get_unchecked(src_row_start + x) };
                            let (r8, g8, b8) = convert(px);

                            // Write RGB directly into dst buffer
                            let di = (dst_y * dst_w + dst_x) * 3;
                            unsafe {
                                // bounds guaranteed by construction
                                *dst_buf.get_unchecked_mut(di + 0) = r8;
                                *dst_buf.get_unchecked_mut(di + 1) = g8;
                                *dst_buf.get_unchecked_mut(di + 2) = b8;
                            }
                        }
                    }
                    // Jump to next candidate on decim grid within the tile
                    // (advance by 1 until aligned, then by decim)
                    if (src_x + 1) % decim == 0 {
                        x += 1;                 // step onto alignment
                    } else {
                        // compute delta to next multiple of decim within the tile band
                        let next = decim - ((src_x + 1) % decim);
                        x += 1 + next;
                    }
                }
            }
        }
    }

    Ok(img)
}

use rayon::prelude::*;
use std::ptr;

/// Mask subtitle region by copying colors from above.
/// Keeps corners intact, only affects the center-bottom area where subtitles appear.
/// - `bottom_frac`: fraction of height for subtitle zone (e.g., 0.12 = bottom 12%)
/// - `corner_frac`: fraction of width to preserve on each side (e.g., 0.15 = outer 15% each side)
pub fn mask_subtitles(img: &mut RgbImage, bottom_frac: f32, corner_frac: f32) {
    let (w, h) = img.dimensions();
    let sub_top = ((1.0 - bottom_frac) * h as f32) as u32;
    let left_edge = (corner_frac * w as f32) as u32;
    let right_edge = ((1.0 - corner_frac) * w as f32) as u32;

    // For each row in the subtitle zone, copy from the row just above the zone
    let src_y = sub_top.saturating_sub(1);
    for y in sub_top..h {
        for x in left_edge..right_edge {
            let px = *img.get_pixel(x, src_y);
            img.put_pixel(x, y, px);
        }
    }
}

pub fn fast_downscale(image: &RgbImage, factor: u32) -> RgbImage {
    let (w, h) = image.dimensions();
    assert!(factor >= 2, "factor must be >= 2");
    assert!(w % factor == 0 && h % factor == 0, "factor must divide dimensions");

    let src_w = w as usize;
    let src_h = h as usize;
    let f = factor as usize;

    let dst_w = src_w / f;
    let dst_h = src_h / f;

    let src = image.as_raw();
    let mut dst = vec![0u8; dst_w * dst_h * 3];

    // Precompute X indices
    let mut x_index = Vec::with_capacity(dst_w);
    for dx in 0..dst_w {
        x_index.push((dx * f) * 3);
    }

    // Make dst_rows mutable so we can use par_iter_mut
    let mut dst_rows: Vec<_> = dst
        .chunks_exact_mut(dst_w * 3)
        .collect();

    dst_rows.par_iter_mut().enumerate().for_each(|(dy, row)| {
        let sy = dy * f;
        let src_row_off = sy * src_w * 3;

        let mut dx = 0;
        while dx + 4 <= dst_w {
            unsafe {
                let s0 = src.as_ptr().add(src_row_off + x_index[dx + 0]);
                let s1 = src.as_ptr().add(src_row_off + x_index[dx + 1]);
                let s2 = src.as_ptr().add(src_row_off + x_index[dx + 2]);
                let s3 = src.as_ptr().add(src_row_off + x_index[dx + 3]);

                let d0 = row.as_mut_ptr().add((dx + 0) * 3);
                let d1 = row.as_mut_ptr().add((dx + 1) * 3);
                let d2 = row.as_mut_ptr().add((dx + 2) * 3);
                let d3 = row.as_mut_ptr().add((dx + 3) * 3);

                ptr::copy_nonoverlapping(s0, d0, 3);
                ptr::copy_nonoverlapping(s1, d1, 3);
                ptr::copy_nonoverlapping(s2, d2, 3);
                ptr::copy_nonoverlapping(s3, d3, 3);
            }
            dx += 4;
        }

        // tail pixels
        while dx < dst_w {
            unsafe {
                let s = src.as_ptr().add(src_row_off + x_index[dx]);
                let d = row.as_mut_ptr().add(dx * 3);
                ptr::copy_nonoverlapping(s, d, 3);
            }
            dx += 1;
        }
    });

    RgbImage::from_raw(dst_w as u32, dst_h as u32, dst)
        .expect("RGB buffer size mismatch")
}

/// Intel modifier for `I915_FORMAT_MOD_Y_TILED_GEN12_RC_CCS_CC`
/// (`0x0100000000000008`). drm-fourcc 2.2.0 has no variant for it, so
/// `DrmModifier::from` produces `Unrecognized(...)`.
const MOD_I915_Y_TILED_GEN12_RC_CCS_CC: u64 = (1u64 << 56) | 8;

/// A GBM device plus the DRM fd it was created from. The device is created on
/// the *render* node when one exists: GBM needs to create a GL/pipe context to
/// blit-decompress render-compressed buffers (see `gbm_dri_bo_map` in Mesa),
/// and context creation reliably works on the render node even for processes
/// that are not DRM master. Using the card node here crashes inside the i915
/// driver (NULL context dereference) when Kodi is the master.
struct GbmContext {
    dev: *mut ffi::gbm::gbm_device,
    /// The DRM fd the device was created from (render node, or card as
    /// fallback). Never read — it is held solely to keep the fd open for the
    /// device's lifetime, since GBM does not duplicate the fd passed to
    /// `gbm_create_device`.
    #[allow(dead_code)]
    fd: RawFd,
}

// SAFETY: The GbmContext owns its device pointer and render-node fd for the
// process lifetime (it lives in a static OnceCell). GBM device operations
// (bo_import/bo_map) are internally synchronised by Mesa, and the pattern
// mirrors `PersistentMap` above: the device pointer is only ever read after
// creation. Sharing between threads is therefore safe.
unsafe impl Send for GbmContext {}
unsafe impl Sync for GbmContext {}

/// Find and open the render node that belongs to the same DRM device as
/// `card`, if one exists. Returns `None` when there is no render node
/// (e.g. Raspberry Pi/VC4), in which case callers fall back to the card fd.
///
/// Each DRM device gets its own character-device major (226 for the first
/// device, 227 for the next, ...); the card node and its render node share
/// that major. So a `renderD*` entry whose `dev` major matches the card's is
/// the sibling render node — no need to dig through the card's sysfs device
/// directory, whose layout varies by kernel.
fn open_render_node(card: &Card) -> Option<RawFd> {
    use std::fs;
    use std::os::unix::fs::MetadataExt;

    // Major of the opened card node (e.g. 226 for /dev/dri/card1).
    let fd_meta = fs::metadata(format!("/proc/self/fd/{}", card.as_raw_fd())).ok()?;
    let card_major = libc::major(fd_meta.rdev());

    let drm_dir = fs::read_dir("/sys/class/drm").ok()?;
    for entry in drm_dir.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let Some(rest) = name.strip_prefix("renderD") else {
            continue;
        };
        if rest.is_empty() || !rest.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }
        let devinfo = match fs::read_to_string(format!("/sys/class/drm/{}/dev", name)) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let mut it = devinfo.trim().split(':');
        let (Ok(maj), Ok(_minor)) =
            (it.next().unwrap_or("").parse::<u32>(), it.next().unwrap_or("").parse::<u32>())
        else {
            continue;
        };
        if maj != card_major {
            continue;
        }
        let path = format!("/dev/dri/{}", name);
        if let Ok(f) = fs::OpenOptions::new().read(true).write(true).open(&path) {
            return Some(f.into_raw_fd());
        }
    }
    None
}

/// Cached GBM device, created lazily. Prefers the render node when one exists
/// (`open_render_node`); falls back to the card fd (no render node, e.g. VC4).
/// The render-node fd is kept open for the device's lifetime because GBM does
/// not duplicate the fd passed to `gbm_create_device`.
static GBM_DEVICE: OnceCell<GbmContext> = OnceCell::new();

/// Import a framebuffer through GBM, map it (detile + decompress) and hand the
/// linear mapping to `f` before unmapping.
///
/// On modern Intel (Gen12+, e.g. Alder Lake-N) Kodi's GBM scanout buffer is
/// render-compressed (`Y_TILED_GEN12_RC_CCS*`). Reading the dma-buf directly
/// yields compressed garbage; `gbm_bo_map` asks the driver to detile and
/// decompress it (via the blitter) so we can read plain linear pixels.
///
/// Processing the mapping in place (rather than copying it into a fresh `Vec`)
/// saves a full-frame allocation plus memcpy on every capture — for 4K that is
/// 16 MB per frame.
fn gbm_with_mapped<R, F>(
    card: &Card,
    fbinfo2: &drm_sys::drm_mode_fb_cmd2,
    verbose: bool,
    f: F,
) -> Result<R, SystemError>
where
    F: FnOnce(&[u32], usize) -> R,
{
    use ffi::gbm;
    use std::ffi::c_void;

    let raw_fd = card.as_raw_fd();

    // The kernel fills handles/pitches/offsets for the first `num_planes`
    // entries; the struct is zero-initialised by `ffi::fb_cmd2`, so any unused
    // trailing entries stay zero.
    let mut num_planes = 0usize;
    for i in 0..4 {
        if fbinfo2.handles[i] != 0 {
            num_planes = i + 1;
        }
    }
    if num_planes == 0 {
        return Err(SystemError::InvalidArgument);
    }

    let mut import = gbm::gbm_import_fd_modifier_data {
        width: fbinfo2.width,
        height: fbinfo2.height,
        format: fbinfo2.pixel_format,
        num_fds: num_planes as u32,
        fds: [0; 4],
        strides: [0; 4],
        offsets: [0; 4],
        modifier: fbinfo2.modifier[0],
    };

    let mut opened = [0i32; 4];
    for i in 0..num_planes {
        let hfd = ffi::prime_handle_to_fd(raw_fd, fbinfo2.handles[i])?;
        import.fds[i] = hfd;
        import.strides[i] = fbinfo2.pitches[i] as i32;
        import.offsets[i] = fbinfo2.offsets[i] as i32;
        opened[i] = hfd;
    }

    let ctx = GBM_DEVICE.get_or_init(|| {
        let fd = open_render_node(card).unwrap_or(raw_fd);
        let dev = unsafe { gbm::gbm_create_device(fd) };
        GbmContext { dev, fd }
    });
    let dev = ctx.dev;
    if dev.is_null() {
        for fd in opened.iter().take(num_planes) {
            unsafe {
                close(*fd);
            }
        }
        return Err(SystemError::Unknown {
            errno: nix::errno::Errno::EINVAL,
        });
    }

    let bo = unsafe {
        gbm::gbm_bo_import(
            dev,
            gbm::GBM_BO_IMPORT_FD_MODIFIER,
            &mut import as *mut _ as *mut c_void,
            gbm::GBM_BO_USE_SCANOUT | gbm::GBM_BO_USE_RENDERING,
        )
    };
    // gbm_bo_import duplicates the fds, so our copies can be closed right away.
    for fd in opened.iter().take(num_planes) {
        unsafe {
            close(*fd);
        }
    }

    if bo.is_null() {
        return Err(SystemError::Unknown {
            errno: nix::errno::Errno::EINVAL,
        });
    }

    let mut stride: u32 = 0;
    let mut map_data: *mut c_void = std::ptr::null_mut();
    let ptr = unsafe {
        gbm::gbm_bo_map(
            bo,
            0,
            0,
            fbinfo2.width,
            fbinfo2.height,
            gbm::GBM_BO_TRANSFER_READ,
            &mut stride,
            &mut map_data,
        )
    };
    if ptr.is_null() {
        unsafe {
            gbm::gbm_bo_destroy(bo);
        }
        return Err(SystemError::Unknown {
            errno: nix::errno::Errno::EINVAL,
        });
    }

    let nwords = (stride as usize * fbinfo2.height as usize) / 4;
    let stride_words = stride as usize / 4;

    // Run the caller's work while the mapping is live (no full-buffer copy).
    let result = unsafe {
        let slice = std::slice::from_raw_parts(ptr as *const u32, nwords);
        f(slice, stride_words)
    };

    unsafe {
        gbm::gbm_bo_unmap(bo, map_data);
        gbm::gbm_bo_destroy(bo);
    }

    if verbose {
        println!(
            "gbm: imported {} plane(s), modifier 0x{:x}, mapped stride {}, {} words",
            num_planes, import.modifier, stride, nwords
        );
    }

    Ok(result)
}

/// Fused decimate + pixel conversion over a linear 32bpp buffer, honouring the
/// source stride (the GBM mapping may be padded).
fn decimate_linear_32<F>(
    size: (u32, u32),
    stride_words: usize,
    src: &[u32],
    decim: usize,
    border_frac: f32,
    convert: F,
) -> RgbImage
where
    F: Fn(u32) -> (u8, u8, u8),
{
    let width = size.0 as usize;
    let height = size.1 as usize;
    let dw = width / decim;
    let dh = height / decim;

    let border_active = border_frac > 0.0;
    let border_top = (height as f32 * border_frac) as usize;
    let border_bot = height.saturating_sub(border_top);
    let border_left = (width as f32 * border_frac) as usize;
    let border_right = width.saturating_sub(border_left);

    let mut img = RgbImage::new(dw as u32, dh as u32);
    let dst = img.as_mut();

    for dy in 0..dh {
        let sy = dy * decim;
        let y_interior = border_active && sy >= border_top && sy < border_bot;
        let src_row = &src[sy * stride_words..];
        let dst_row = &mut dst[dy * dw * 3..dy * dw * 3 + dw * 3];
        for dx in 0..dw {
            let sx = dx * decim;
            if y_interior && sx >= border_left && sx < border_right {
                continue;
            }
            // Bounds are structural: sx < width <= stride_words, and the writes
            // stay inside the 3-wide slice for this row.
            let px = unsafe { *src_row.get_unchecked(sx) };
            let (r, g, b) = convert(px);
            let o = dx * 3;
            unsafe {
                *dst_row.get_unchecked_mut(o) = r;
                *dst_row.get_unchecked_mut(o + 1) = g;
                *dst_row.get_unchecked_mut(o + 2) = b;
            }
        }
    }

    img
}

/// Fused decimate + 10-bit ↔ 8-bit conversion for the GBM path, specialized for
/// `XRGB2101010`/`ARGB2101010` sources.
///
/// This mirrors `decimate_linear_32` with `ten_to_eight` as the converter, but:
/// - all tone-mapping state (PQ vs SDR, LUT, luminance, saturation) is resolved
///   once per frame instead of once per pixel with atomic loads and a
///   `OnceCell` check, and
/// - rows are independent, so work is spread across all cores with rayon
///   (the same pattern `fast_downscale` already uses).
///
/// Output is identical to the generic path.
fn decimate_linear_32_tenbit(
    size: (u32, u32),
    stride_words: usize,
    src: &[u32],
    decim: usize,
    border_frac: f32,
) -> RgbImage {
    let width = size.0 as usize;
    let height = size.1 as usize;
    let dw = width / decim;
    let dh = height / decim;

    // Resolve the conversion state once, before the pixel loops.
    let hdr = HDR_PQ_MODE.load(Ordering::Relaxed);
    let (lut, lum, sat) = if hdr {
        (
            pq_channel_lut(),
            f32::from_bits(LUMINANCE_HDR.load(Ordering::Relaxed)),
            f32::from_bits(SATURATION_HDR.load(Ordering::Relaxed)),
        )
    } else {
        (
            sdr_channel_lut(),
            f32::from_bits(LUMINANCE_SDR.load(Ordering::Relaxed)),
            0.0,
        )
    };
    let scale = lum * 255.0;

    let border_active = border_frac > 0.0;
    let border_top = (height as f32 * border_frac) as usize;
    let border_bot = height.saturating_sub(border_top);
    let border_left = (width as f32 * border_frac) as usize;
    let border_right = width.saturating_sub(border_left);

    let mut img = RgbImage::new(dw as u32, dh as u32);
    let dst = img.as_mut();
    let mut dst_rows: Vec<&mut [u8]> = dst.chunks_exact_mut(dw * 3).collect();

    dst_rows.par_iter_mut().enumerate().for_each(|(dy, row)| {
        let sy = dy * decim;
        let src_row = &src[sy * stride_words..];
        let interior = border_active && sy >= border_top && sy < border_bot;
        if hdr {
            for dx in 0..dw {
                let sx = dx * decim;
                if interior && sx >= border_left && sx < border_right {
                    continue;
                }
                let px = unsafe { *src_row.get_unchecked(sx) };
                let r = unsafe { *lut.get_unchecked(((px >> 20) & 0x3FF) as usize) };
                let g = unsafe { *lut.get_unchecked(((px >> 10) & 0x3FF) as usize) };
                let b = unsafe { *lut.get_unchecked((px & 0x3FF) as usize) };
                // PQ decode + tone map + gamma come from the LUT; only the
                // cross-channel saturation and luminance scaling remain.
                let avg = (r + g + b) / 3.0;
                let (r, g, b) = (
                    (avg + (r - avg) * sat) * scale,
                    (avg + (g - avg) * sat) * scale,
                    (avg + (b - avg) * sat) * scale,
                );
                let o = dx * 3;
                unsafe {
                    *row.get_unchecked_mut(o) = r.clamp(0.0, 255.0) as u8;
                    *row.get_unchecked_mut(o + 1) = g.clamp(0.0, 255.0) as u8;
                    *row.get_unchecked_mut(o + 2) = b.clamp(0.0, 255.0) as u8;
                }
            }
        } else {
            for dx in 0..dw {
                let sx = dx * decim;
                if interior && sx >= border_left && sx < border_right {
                    continue;
                }
                let px = unsafe { *src_row.get_unchecked(sx) };
                let r = unsafe { *lut.get_unchecked(((px >> 20) & 0x3FF) as usize) };
                let g = unsafe { *lut.get_unchecked(((px >> 10) & 0x3FF) as usize) };
                let b = unsafe { *lut.get_unchecked((px & 0x3FF) as usize) };
                let (r, g, b) = (r * scale, g * scale, b * scale);
                let o = dx * 3;
                unsafe {
                    *row.get_unchecked_mut(o) = r.clamp(0.0, 255.0) as u8;
                    *row.get_unchecked_mut(o + 1) = g.clamp(0.0, 255.0) as u8;
                    *row.get_unchecked_mut(o + 2) = b.clamp(0.0, 255.0) as u8;
                }
            }
        }
    });

    img
}

/// Decode an Intel Y-tiled / render-compressed framebuffer by round-tripping it
/// through GBM (detile + decompress) and then reading it linearly.
fn dump_intel_gbm_to_image(
    card: &Card,
    fbinfo2: &drm_sys::drm_mode_fb_cmd2,
    size: (u32, u32),
    ten_bit: bool,
    border_frac: f32,
    verbose: bool,
) -> Result<RgbImage, SystemError> {
    let decim = pick_decim(size.0, size.1, 6);

    if ten_bit {
        // Specialized 10-bit path: conversion state resolved once per frame and
        // rows processed in parallel.
        gbm_with_mapped(card, fbinfo2, verbose, |buf, stride_words| {
            decimate_linear_32_tenbit(size, stride_words, buf, decim, border_frac)
        })
    } else {
        let convert = move |px: u32| -> (u8, u8, u8) {
            (
                ((px >> 16) & 0xff) as u8,
                ((px >> 8) & 0xff) as u8,
                (px & 0xff) as u8,
            )
        };

        // Decimate straight out of the GBM mapping instead of copying it first.
        gbm_with_mapped(card, fbinfo2, verbose, |buf, stride_words| {
            decimate_linear_32(
                size,
                stride_words,
                buf,
                decim,
                border_frac,
                convert,
            )
        })
    }
}

pub fn dump_framebuffer_to_image(
    card: &Card,
    fb: Handle,
    verbose: bool,
    mask_subs: bool,
    border_frac: f32,
) -> Result<RgbImage, SystemError> {
    let fbinfo2 = ffi::fb_cmd2(card.as_raw_fd(), fb.into())?;

    if verbose {
        println!("  -> FB Info 2: {:?}", fbinfo2);
    }

    let size = (fbinfo2.width, fbinfo2.height);

    if fbinfo2.pixel_format == 808661072 {
        return decode_p030_image(
            card,
            (size.0 as _, size.1 as _),
            fbinfo2.pitches[0],
            fbinfo2.handles[0],
            fbinfo2.modifier[0],
            fbinfo2.offsets[1] as _,
            verbose,
        );
    }

    let fourcc = drm_fourcc::DrmFourcc::try_from(fbinfo2.pixel_format).unwrap();
    // Use `From` so unrecognized modifiers become `Unrecognized(..)` and can be
    // matched explicitly (drm-fourcc 2.2.0 has no variant for
    // `Y_TILED_GEN12_RC_CCS_CC`).
    let modifier = drm_fourcc::DrmModifier::from(fbinfo2.modifier[0]);

    let image_result = match fourcc {
        DrmFourcc::Xrgb8888 | DrmFourcc::Argb8888 => match modifier {
            DrmModifier::Broadcom_vc4_t_tiled => {
                dump_broadcom_tiled_to_image(card, size, 32, fbinfo2.handles[0], verbose)
            }
            DrmModifier::Linear => dump_linear_to_image(
                card,
                fbinfo2.pitches[0],
                size,
                32,
                fbinfo2.handles[0],
                border_frac,
                verbose,
            ),
            DrmModifier::I915_x_tiled => dump_intel_xtiled_decimated_to_image(
                card,
                fbinfo2.pitches[0],
                size,
                fbinfo2.handles[0],
                pick_decim(size.0, size.1, 6),
                border_frac,
                verbose,
                |px| (((px >> 16) & 0xff) as u8, ((px >> 8) & 0xff) as u8, (px & 0xff) as u8),
            ),
            // Y-tiled and Gen12 render-compressed (RC/MC CCS) buffers are read
            // back via GBM, which detiles and decompresses them.
            DrmModifier::I915_y_tiled
            | DrmModifier::I915_y_tiled_ccs
            | DrmModifier::I915_y_tiled_gen12_mc_ccs
            | DrmModifier::I915_y_tiled_gen12_rc_ccs
            | DrmModifier::Unrecognized(MOD_I915_Y_TILED_GEN12_RC_CCS_CC) => {
                dump_intel_gbm_to_image(card, &fbinfo2, size, false, border_frac, verbose)
            }
            _ => panic!("Unsupported framebuffer modifier: {:?}", modifier),
        },
        // Kodi prefers an EGL config with an alpha channel, so Intel scanout
        // buffers are usually AR30/AR24 rather than XR30/XR24. Both share the
        // same R/G/B bit layout, so decode them together.
        DrmFourcc::Xrgb2101010 | DrmFourcc::Argb2101010 => match modifier {
            DrmModifier::Linear => dump_linear_xr30_to_image(
                card,
                fbinfo2.pitches[0],
                size,
                32,
                fbinfo2.handles[0],
                border_frac,
                verbose,
            ),
            DrmModifier::I915_x_tiled => dump_intel_xtiled_decimated_to_image(
                card,
                fbinfo2.pitches[0],
                size,
                fbinfo2.handles[0],
                pick_decim(size.0, size.1, 6),
                border_frac,
                verbose,
                |px| {
                    let (r, g, b) = ten_to_eight(px);
                    (r as u8, g as u8, b as u8)
                },
            ),
            // Y-tiled and Gen12 render-compressed (RC/MC CCS) buffers: read
            // back via GBM so the driver detiles/decompresses them.
            DrmModifier::I915_y_tiled
            | DrmModifier::I915_y_tiled_ccs
            | DrmModifier::I915_y_tiled_gen12_mc_ccs
            | DrmModifier::I915_y_tiled_gen12_rc_ccs
            | DrmModifier::Unrecognized(MOD_I915_Y_TILED_GEN12_RC_CCS_CC) => {
                dump_intel_gbm_to_image(card, &fbinfo2, size, true, border_frac, verbose)
            }
            // Any other modifier: assume the buffer is linear.
            _ => dump_linear_xr30_to_image(
                card,
                fbinfo2.pitches[0],
                size,
                32,
                fbinfo2.handles[0],
                border_frac,
                verbose,
            ),
        },
        DrmFourcc::Yuv420 => dump_yuv420_to_image(
            card,
            size,
            fbinfo2.pitches,
            fbinfo2.handles,
            fbinfo2.offsets,
            verbose,
        ),
        DrmFourcc::Rgb565 => dump_rgb565_to_image(
            card,
            fbinfo2.pitches[0],
            size,
            16,
            fbinfo2.handles[0],
            verbose,
        ),
        DrmFourcc::Nv12 => decode_nv12_image(
            card,
            (size.0 as _, size.1 as _),
            fbinfo2.pitches[0],
            fbinfo2.handles[0],
            fbinfo2.modifier[0],
            fbinfo2.offsets[1] as _,
            verbose,
        ),

        _ => panic!(
            "Unsupported framebuffer pixel format: {} {:x}",
            fourcc, fbinfo2.pixel_format
        ),
    };

    if let Err(e) = gem_close(card.as_raw_fd(), fbinfo2.handles[0]) {
        match e {
            SystemError::InvalidArgument => { /* ignore EINVAL */ }
            SystemError::Unknown { errno } if errno == nix::errno::Errno::ENOENT => { /* ignore ENOENT */ }
            other => eprintln!("gem_close failed: {:?}", other),
        }
    }

    let mut image = image_result?;
    if mask_subs {
        mask_subtitles(&mut image, 0.20, 0.15);
    }
    Ok(image)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth(width: usize, height: usize) -> Vec<u32> {
        // Deterministic pseudo-random 10-bit packed pixels.
        (0..width * height)
            .map(|i| {
                let x = i as u32;
                ((x.wrapping_mul(2654435761) & 0x3FF))
                    | ((x.wrapping_mul(40503) & 0x3FF) << 10)
                    | ((x.wrapping_mul(2246822519) & 0x3FF) << 20)
            })
            .collect()
    }

    /// The specialized 10-bit decimate must be bit-identical to the generic
    /// `decimate_linear_32` + `ten_to_eight` path, in both HDR and SDR mode and
    /// with the border mask on and off.
    #[test]
    fn tenbit_decimate_matches_generic() {
        let (width, height) = (3840usize, 2160usize);
        let src = synth(width, height);
        let size = (width as u32, height as u32);
        let decim = pick_decim(size.0, size.1, 6);

        for &hdr in &[true, false] {
            HDR_PQ_MODE.store(hdr, Ordering::Relaxed);
            for &border in &[0.0f32, 0.1] {
                let generic = decimate_linear_32(
                    size,
                    width,
                    &src,
                    decim,
                    border,
                    |px| {
                        let (r, g, b) = ten_to_eight(px);
                        (r as u8, g as u8, b as u8)
                    },
                );
                let specialized =
                    decimate_linear_32_tenbit(size, width, &src, decim, border);
                assert_eq!(
                    generic.as_raw(),
                    specialized.as_raw(),
                    "hdr={} border={}",
                    hdr,
                    border
                );
            }
        }
    }
}
