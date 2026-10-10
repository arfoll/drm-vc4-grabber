use drm_ffi::result::SystemError;
use drm_sys::drm_mode_fb_cmd2;
use drm_sys::*;
use std::os::unix::prelude::RawFd;

ioctl_readwrite!(drm_mode_getfb2, DRM_IOCTL_BASE, 0xCE, drm_mode_fb_cmd2);

pub fn fb_cmd2(fd: RawFd, handle: u32) -> Result<drm_mode_fb_cmd2, SystemError> {
    let mut fb = drm_mode_fb_cmd2 {
        fb_id: handle,
        width: 0,
        height: 0,
        pixel_format: 0,
        flags: 0,
        handles: [0; 4],
        pitches: [0; 4],
        offsets: [0; 4],
        modifier: [0; 4],
    };

    unsafe {
        drm_mode_getfb2(fd, &mut fb)?;
    }

    Ok(fb)
}

ioctl_readwrite!(drm_prime_handle_to_fd, DRM_IOCTL_BASE, 0x2D, drm_prime_handle);

pub fn prime_handle_to_fd(fd: RawFd, handle: u32) -> Result<RawFd, SystemError> {
    let mut ph = drm_prime_handle {
        handle,
        flags: 0,
        fd: 0,
    };

    unsafe {
        drm_prime_handle_to_fd(fd, &mut ph)?;
    }

    Ok(ph.fd)
}

ioctl_write_ptr!(drm_gem_close_ioctl, DRM_IOCTL_BASE, 0x09, drm_gem_close);

pub fn gem_close(fd: RawFd, handle: u32) -> Result<(), SystemError> {
    let mut ph = drm_gem_close {
        handle,
        pad: 0,
    };

    unsafe {
        drm_gem_close_ioctl(fd, &mut ph)?;
    }

    Ok(())
}

/// Minimal bindings to libgbm. We use GBM only to import an existing
/// framebuffer (as a dma-buf with its format modifier) and map it back to CPU
/// memory. For tiled/compressed Intel buffers `gbm_bo_map` transparently
/// detiles/decompresses via the driver, which is the only reliable way to read
/// render-compressed (RC_CCS) scanout buffers from userspace.
pub mod gbm {
    use std::ffi::c_void;
    use std::os::unix::io::RawFd;

    #[repr(C)]
    pub struct gbm_device {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct gbm_bo {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct gbm_import_fd_modifier_data {
        pub width: u32,
        pub height: u32,
        pub format: u32,
        pub num_fds: u32,
        pub fds: [i32; 4],
        pub strides: [i32; 4],
        pub offsets: [i32; 4],
        pub modifier: u64,
    }

    pub const GBM_BO_IMPORT_FD_MODIFIER: u32 = 0x5504;
    pub const GBM_BO_USE_SCANOUT: u32 = 1 << 0;
    pub const GBM_BO_USE_RENDERING: u32 = 1 << 2;
    pub const GBM_BO_TRANSFER_READ: u32 = 1 << 0;

    #[link(name = "gbm")]
    extern "C" {
        pub fn gbm_create_device(fd: RawFd) -> *mut gbm_device;
        pub fn gbm_device_destroy(dev: *mut gbm_device);
        pub fn gbm_bo_import(
            dev: *mut gbm_device,
            type_: u32,
            buffer: *mut c_void,
            flags: u32,
        ) -> *mut gbm_bo;
        pub fn gbm_bo_map(
            bo: *mut gbm_bo,
            x: u32,
            y: u32,
            width: u32,
            height: u32,
            flags: u32,
            stride: *mut u32,
            map_data: *mut *mut c_void,
        ) -> *mut c_void;
        pub fn gbm_bo_unmap(bo: *mut gbm_bo, map_data: *mut c_void);
        pub fn gbm_bo_destroy(bo: *mut gbm_bo);
    }
}
