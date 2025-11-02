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

fn dump_intel_xtiled_xr30_to_image(
    card: &Card,
    pitch: u32,
    size: (u32, u32),
    handle: u32,
    verbose: bool,
) -> Result<RgbImage, SystemError> {
    // Intel X-tiled 32bpp: 128×8 pixels per tile = 4096 bytes
    const TILE_W: usize = 128;
    const TILE_H: usize = 8;
    const TILE_BYTES: usize = 4096;

    let width  = size.0 as usize;
    let height = size.1 as usize;
    let pitch_words = pitch as usize / 4;

    if verbose {
        println!(
            "intel x-tiled XR30 (final), size={:?}, pitch={}, pitch_words={}",
            size, pitch, pitch_words
        );
    }

    let mut copy = vec![0u32; pitch_words * height];
    copy_buffer(card, handle, &mut copy, verbose)?;
    let mut img = RgbImage::new(size.0, size.1);

    let tiles_x = (width + TILE_W - 1) / TILE_W;
    let tiles_y = (height + TILE_H - 1) / TILE_H;
    let tiles_per_pitch = pitch as usize / (TILE_W * 4);

    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let tile_base = (ty * tiles_per_pitch + tx) * (TILE_BYTES / 4);

            for y in 0..TILE_H {
                let dst_y = ty * TILE_H + y;
                if dst_y >= height { break; }

                let dst_x = tx * TILE_W;
                let src_start = tile_base + y * (TILE_W);
                let src_end = (src_start + TILE_W).min(copy.len());
                if src_start >= copy.len() { break; }

                for (i, &px) in copy[src_start..src_end].iter().enumerate() {
                    if dst_x + i >= width { break; }
                    let (r8, g8, b8) = ten_to_eight(px);
                    img.put_pixel(
                        (dst_x + i) as u32,
                        dst_y as u32,
                        image::Rgb([r8 as u8, g8 as u8, b8 as u8]),
                    );
                }
            }
        }
    }

    Ok(img)
}

fn fast_downscale(img: &RgbImage, factor: u32) -> RgbImage {
    let (w, h) = img.dimensions();
    let new_w = w / factor;
    let new_h = h / factor;
    let mut out = RgbImage::new(new_w, new_h);

    // Sample one pixel per block (very fast)
    for y in 0..new_h {
        for x in 0..new_w {
            let px = img.get_pixel(x * factor, y * factor);
            out.put_pixel(x, y, *px);
        }
    }
    out
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
        dump_intel_xtiled_xr30_to_image(
            card, fbinfo2.pitches[0], size, fbinfo2.handles[0], verbose,
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
    let scaled = fast_downscale(&image, 10);
    Ok(scaled)
}
