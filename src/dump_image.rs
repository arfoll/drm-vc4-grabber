use std::{convert::TryFrom, mem::size_of, os::fd::AsRawFd};

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

use std::os::fd::RawFd;
use nix::sys::mman::{ProtFlags, MapFlags};
use once_cell::sync::OnceCell;
use std::collections::HashMap;
use std::sync::Mutex;

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

fn decimate_image_4(size: (usize, usize), image: &[u32], copy: &mut [u32]) {
    let decim = (4, 4);
    let newsize = (size.0 / decim.0, size.1 / decim.1);

    for y in 0..newsize.1 {
        let ty = decim.1 * y;
        for x in 0..newsize.0 {
            let tx = decim.0 * x;
            copy[y * newsize.0 + x] = image[ty * size.0 + tx];
        }
    }
}

/// Convert in-place from XRGB2101010/ARGB2101010 to XRGB8888 order.
/// Input: 0b XX RRRRRRRRRR GGGGGGGGGG BBBBBBBBBB (bits: 31..0)
/// Output (LE memory): u32 with bytes [BB, GG, RR, 00]
#[inline]
fn ten_to_eight(v10: u32) -> (u32, u32, u32) {
    // Scale 10->8 with rounding: (x*255 + 511)/1023
    let scale = |x: u32| ((x * 255 + 511) / 1023) & 0xFF;
    let r10 = (v10 >> 20) & 0x3FF;
    let g10 = (v10 >> 10) & 0x3FF;
    let b10 = (v10 >>  0) & 0x3FF;
    (scale(r10), scale(g10), scale(b10))
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
    verbose: bool,
) -> Result<RgbImage, SystemError> {
    let size = (size.0, size.1);

    let length = pitch * size.1 / (bpp / 8);

    println!(
        "linear, size: {:?}, pitch: {}, bpp: {}, length: {}",
        size, pitch, bpp, length
    );
    let mut copy = vec![0u32; length as _];
    copy_buffer(card, handle, &mut copy, verbose)?;

    let mut dec = vec![0u32; (length / (4 * 4)) as _];
    decimate_image_4(
        (size.0 as _, size.1 as _),
        copy.as_slice(),
        dec.as_mut_slice(),
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
    verbose: bool,
) -> Result<RgbImage, SystemError> {
    // Even though it's 30bpp, DRM buffers are 32-bit words per pixel.
    let length_words = (pitch * size.1) / 4;
    println!(
        "linear XR30, size: {:?}, pitch: {}, words: {}",
        size, pitch, length_words
    );

    let mut copy = vec![0u32; length_words as _];
    copy_buffer(card, handle, &mut copy, verbose)?;

    // Convert XR30→XR24 in place
    xr30_to_xr24_inplace(&mut copy);

    // Decimate to keep CPU work low (like the XR24 path)
    let mut dec = vec![0u32; (length_words / (4 * 4)) as _];
    decimate_image_4((size.0 as _, size.1 as _), copy.as_slice(), dec.as_mut_slice());

    Ok(decode_image(
        dec.as_mut_slice(),
        pitch / 4,
        (size.0 / 4, size.1 / 4),
    ))
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

    println!(
        "rgb565, size: {:?}, pitch: {}, bpp: {}, length: {}",
        size, pitch, bpp, length
    );
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

/// Fused untile + decimate (nearest) for Intel X-tiled XR30.
/// - Input: XR30 (10-bit per channel packed into 32bpp words), Intel X-tiling modifier ((1<<56)|1)
/// - Output: RgbImage downscaled by integer `decim` (e.g. 6 → 3840x2160 → 640x360)
fn dump_intel_xtiled_xr30_decimated_to_image(
    card: &Card,
    pitch: u32,
    size: (u32, u32),
    handle: u32,
    decim: usize,
    verbose: bool,
) -> Result<image::RgbImage, SystemError> {
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
            "intel x-tiled XR30 (fused decimate), src={:?}, pitch_words={}, tiles=({}x{}), decim={}",
            size, pitch_words, tiles_x, tiles_y, decim
        );
    }

    // Destination image (RGB888)
    let mut img = image::RgbImage::new(dst_w as u32, dst_h as u32);
    let dst_buf = img.as_mut(); // &mut [u8]

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

            for tx in 0..tiles_x {
                let tile_x0 = tx * TILE_W;

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
                        let dst_x = src_x / decim;

                        // Fetch XR30 pixel (u32), convert 10→8
                        let px10 = unsafe { *src.get_unchecked(src_row_start + x) };
                        let (r8, g8, b8) = ten_to_eight(px10);

                        // Write RGB directly into dst buffer
                        let di = (dst_y * dst_w + dst_x) * 3;
                        unsafe {
                            // bounds guaranteed by construction
                            *dst_buf.get_unchecked_mut(di + 0) = r8 as u8;
                            *dst_buf.get_unchecked_mut(di + 1) = g8 as u8;
                            *dst_buf.get_unchecked_mut(di + 2) = b8 as u8;
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

pub fn dump_framebuffer_to_image(
    card: &Card,
    fb: Handle,
    verbose: bool,
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
    let modifier = drm_fourcc::DrmModifier::try_from(fbinfo2.modifier[0]).unwrap();

    let image_result = match fourcc {
        DrmFourcc::Xrgb8888 => match modifier {
            DrmModifier::Broadcom_vc4_t_tiled => {
                dump_broadcom_tiled_to_image(card, size, 32, fbinfo2.handles[0], verbose)
            }
            DrmModifier::Linear => dump_linear_to_image(
                card,
                fbinfo2.pitches[0],
                size,
                32,
                fbinfo2.handles[0],
                verbose,
            ),
            _ => panic!("Unsupported framebuffer modifier: {:?}", modifier),
        },
        DrmFourcc::Argb8888 => match modifier {
            DrmModifier::Broadcom_vc4_t_tiled => {
                dump_broadcom_tiled_to_image(card, size, 32, fbinfo2.handles[0], verbose)
            }
            DrmModifier::Linear => dump_linear_to_image(
                card,
                fbinfo2.pitches[0],
                size,
                32,
                fbinfo2.handles[0],
                verbose,
            ),
            _ => panic!("Unsupported framebuffer modifier: {:?}", modifier),
        },
        DrmFourcc::Xrgb2101010 => {
        if fbinfo2.modifier[0] == ((1u64 << 56) | 1) {
            // Choose your integer factor. 6 → 640x360 from 3840x2160.
            let decim = 6usize;
                dump_intel_xtiled_xr30_decimated_to_image(
                  card, fbinfo2.pitches[0], size, fbinfo2.handles[0], decim, verbose,
                )
            } else {
                dump_linear_xr30_to_image(
                    card, fbinfo2.pitches[0], size, 32, fbinfo2.handles[0], verbose,
                )
            }
        }
        DrmFourcc::Argb2101010 => {
            dump_linear_xr30_to_image(
                card,
                fbinfo2.pitches[0],
                size,
                32,
                fbinfo2.handles[0],
                verbose,
            )
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

    let image = image_result?;
    Ok(image)
}
