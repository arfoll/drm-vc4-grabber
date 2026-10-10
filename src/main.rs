#[macro_use]
extern crate nix;

use std::fs::{File, OpenOptions};
use std::net::{Shutdown, TcpStream};
use std::os::fd::AsFd;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;

use clap::{App, Arg};
use drm::control::framebuffer::Handle;
use drm::control::{Device as ControlDevice, connector};
use drm::Device;
use drm_ffi::drm_set_client_cap;

use dump_image::{dump_framebuffer_to_image, sample_framebuffer, set_hdr_pq_mode, set_luminance_hdr, set_luminance_sdr, set_saturation_hdr};
use image::{ImageError, RgbImage};

use std::os::unix::io::{AsRawFd, RawFd};
use std::{thread, time::Duration};
use std::time::Instant;

use std::io::Result as StdResult;

pub mod ffi;
pub mod framebuffer;
pub mod hyperion;
pub mod hyperion_reply_generated;
pub mod hyperion_request_generated;
pub mod image_decoder;
pub mod dump_image;

pub use hyperion_request_generated::hyperionnet::{Clear, Color, Command, Image, Register};

use hyperion::{read_reply, register_direct, send_color_red, send_image};

pub struct Card(File);

impl AsRawFd for Card {
    fn as_raw_fd(&self) -> RawFd {
        self.0.as_raw_fd()
    }
}

impl AsFd for Card {
    fn as_fd(&self) -> std::os::fd::BorrowedFd<'_> {
        self.0.as_fd()
    }
}

impl Device for Card {}
impl ControlDevice for Card {}

impl Card {
    pub fn open(path: &str) -> Self {
        let mut options = OpenOptions::new();
        options.read(true);
        options.write(false);
        Card(options.open(path).unwrap())
    }
}

fn save_screenshot(img: &RgbImage) -> Result<(), ImageError> {
    img.save("screenshot.png")
}

/// How long to wait between reconnection attempts after the server goes away.
const RECONNECT_INTERVAL: Duration = Duration::from_secs(3);

/// A live connection to the Hyperion server.
///
/// The server sends a reply for every request. Images don't wait for them (that
/// would serialise every frame behind a round trip), so a background thread
/// drains them. If either the reader or a write fails, the connection is marked
/// dead and the caller replaces it.
struct Connection {
    stream: TcpStream,
    alive: Arc<AtomicBool>,
    drain: Option<JoinHandle<()>>,
}

impl Connection {
    /// Connect, register, and start draining replies. `show_color` reproduces
    /// the original startup red flash; reconnects skip it to avoid flashing the
    /// LEDs mid-playback.
    fn connect(address: &str, verbose: bool, show_color: bool) -> StdResult<Connection> {
        let mut stream = TcpStream::connect(address)?;
        // Hyperion frames messages as a 4-byte length plus payload, sent as two
        // small writes, each followed by a small reply. With Nagle enabled this
        // is a classic Nagle + delayed-ACK stall (~40ms per round trip).
        stream.set_nodelay(true)?;

        // Register before the drain thread starts reading, so it does not
        // consume the registration confirmation.
        register_direct(&mut stream)?;
        read_reply(&mut stream, verbose)?;

        if show_color {
            send_color_red(&mut stream, verbose)?;
        }

        let alive = Arc::new(AtomicBool::new(true));
        let drain_alive = Arc::clone(&alive);
        let mut drain_stream = stream.try_clone()?;
        let drain = thread::spawn(move || {
            while read_reply(&mut drain_stream, false).is_ok() {}
            drain_alive.store(false, Ordering::SeqCst);
        });

        Ok(Connection {
            stream,
            alive,
            drain: Some(drain),
        })
    }

    fn is_alive(&self) -> bool {
        self.alive.load(Ordering::SeqCst)
    }

    /// Send one image. On failure the connection is marked dead so the caller
    /// reconnects.
    fn send_image(&mut self, image: &RgbImage, verbose: bool) -> StdResult<()> {
        let result = send_image(&mut self.stream, image, verbose);
        if result.is_err() {
            self.alive.store(false, Ordering::SeqCst);
        }
        result
    }
}

impl Drop for Connection {
    fn drop(&mut self) {
        // Wake the drain thread out of its blocking read, then reap it so
        // reconnects don't leak a thread each time.
        let _ = self.stream.shutdown(Shutdown::Both);
        if let Some(handle) = self.drain.take() {
            let _ = handle.join();
        }
    }
}

/// Send one frame, reconnecting (rate-limited) if the connection is down.
/// Returns true if the frame was delivered.
fn send_frame(
    conn: &mut Option<Connection>,
    address: &str,
    image: &RgbImage,
    verbose: bool,
    last_attempt: &mut Instant,
) -> bool {
    if conn.is_none() {
        if last_attempt.elapsed() < RECONNECT_INTERVAL {
            return false;
        }
        *last_attempt = Instant::now();
        match Connection::connect(address, verbose, false) {
            Ok(connection) => {
                println!(">>> Reconnected to Hyperion at {}", address);
                *conn = Some(connection);
            }
            Err(e) => {
                if verbose {
                    eprintln!("Hyperion reconnect failed: {}", e);
                }
                return false;
            }
        }
    }

    let connection = match conn.as_mut() {
        Some(connection) => connection,
        None => return false,
    };

    if !connection.is_alive() {
        if verbose {
            eprintln!("Hyperion connection closed; will reconnect.");
        }
        *conn = None;
        return false;
    }

    match connection.send_image(image, verbose) {
        Ok(()) => true,
        Err(e) => {
            eprintln!("Hyperion send failed ({}); will reconnect.", e);
            *conn = None;
            false
        }
    }
}

/// Check if HDR is active by examining connector properties.
/// Returns true if Colorspace is BT2020 (values 8-10) or if HDR_OUTPUT_METADATA is set.
fn detect_hdr_mode(card: &Card, verbose: bool) -> bool {
    let resource_handles = match card.resource_handles() {
        Ok(h) => h,
        Err(_) => return false,
    };

    for conn_handle in resource_handles.connectors() {
        let conn_info = match card.get_connector(*conn_handle) {
            Ok(info) => info,
            Err(_) => continue,
        };

        // Only check connected connectors
        if conn_info.state() != connector::State::Connected {
            continue;
        }

        // Get properties for this connector
        let props = match card.get_properties(*conn_handle) {
            Ok(p) => p,
            Err(_) => continue,
        };

        let (handles, values) = props.as_props_and_values();
        for (prop_handle, &value) in handles.iter().zip(values.iter()) {
            let prop_info = match card.get_property(*prop_handle) {
                Ok(info) => info,
                Err(_) => continue,
            };

            let name = prop_info.name().to_str().unwrap_or("");

            if name == "Colorspace" {
                if verbose {
                    println!("Colorspace={}", value);
                }
                // BT2020_CYCC=8, BT2020_RGB=9, BT2020_YCC=10
                if value >= 8 && value <= 10 {
                    if verbose {
                        println!("HDR detected: Colorspace is BT2020");
                    }
                    return true;
                }
            }

            if name == "HDR_OUTPUT_METADATA" {
                if verbose {
                    println!("HDR_OUTPUT_METADATA blob_id={}", value);
                }
                // Non-zero blob ID means HDR metadata is set
                // But we need to verify the blob actually has data
                if value != 0 {
                    if let Ok(blob_data) = card.get_property_blob(value) {
                        if !blob_data.is_empty() {
                            if verbose {
                                println!("HDR detected: HDR_OUTPUT_METADATA has {} bytes", blob_data.len());
                            }
                            return true;
                        }
                    }
                }
            }
        }
    }

    false
}

fn find_framebuffer(card: &Card, verbose: bool) -> Option<Handle> {
    let resource_handles = card.resource_handles().unwrap();

    for crtc in resource_handles.crtcs() {
        let info = card.get_crtc(*crtc).unwrap();

        if verbose {
            println!("CRTC Info: {:?}", info);
        }

        if info.mode().is_some() {
            if let Some(fb) = info.framebuffer() {
                return Some(fb);
            }
        }
    }

    let plane_handles = card.plane_handles().unwrap();

    for plane in plane_handles.planes() {
        let info = card.get_plane(*plane).unwrap();

        if verbose {
            println!("Plane Info: {:?}", info);
        }

        if info.crtc().is_some() {
            let fb = info.framebuffer().unwrap();

            return Some(fb);
        }
    }

    None
}

fn main() {
    let matches = App::new("DRM VC4 Screen Grabber for Hyperion")
        .version("0.1.0")
        .author("Rudi Horn <dyn-git@rudi-horn.de>")
        .about("Captures a screenshot and sends it to the Hyperion server.")
        .arg(
            Arg::with_name("device")
                .short("d")
                .long("device")
                .default_value("/dev/dri/card0")
                .takes_value(true)
                .help("The device path of the DRM device to capture the image from."),
        )
        .arg(
            Arg::with_name("address")
                .short("a")
                .long("address")
                .default_value("127.0.0.1:19400")
                .takes_value(true)
                .help("The Hyperion TCP socket address to send the captured screenshots to."),
        )
        .arg(
            Arg::with_name("screenshot")
                .long("screenshot")
                .takes_value(false)
                .help("Capture a screenshot and save it to screenshot.png"),
        )
        .arg(
            Arg::with_name("verbose")
                .short("v")
                .long("verbose")
                .help("Print verbose debugging information."),
        )
        .arg(
            Arg::with_name("mask-subtitles")
                .short("m")
                .long("mask-subtitles")
                .help("Mask the subtitle region (bottom center) to avoid color flicker."),
        )
        .arg(
            Arg::with_name("hdr-pq")
                .long("hdr-pq")
                .help("Force PQ HDR tone mapping on (auto-detected by default)."),
        )
        .arg(
            Arg::with_name("no-hdr")
                .long("no-hdr")
                .help("Disable HDR tone mapping (use simple power curve)."),
        )
        .arg(
            Arg::with_name("luminance-hdr")
                .long("luminance-hdr")
                .takes_value(true)
                .default_value("1.0")
                .help("Luminance multiplier for HDR content (default 1.0)."),
        )
        .arg(
            Arg::with_name("luminance-sdr")
                .long("luminance-sdr")
                .takes_value(true)
                .default_value("1.0")
                .help("Luminance multiplier for SDR 10-bit content (default 1.0)."),
        )
        .arg(
            Arg::with_name("saturation")
                .short("s")
                .long("saturation")
                .takes_value(true)
                .default_value("1.3")
                .help("Saturation boost for HDR content (default 1.3, compensates for BT.2020 gamut)."),
        )
        .arg(
            Arg::with_name("border")
                .short("b")
                .long("border")
                .takes_value(true)
                .default_value("0")
                .help("Edge-only capture: fraction of each edge to process (e.g. 0.2 = outer 20%). Skips interior pixels to reduce CPU. Use >= 0.15 to clear black bars."),
        )
        .arg(
            Arg::with_name("skip-unchanged")
                .long("skip-unchanged")
                .help("Skip frames that haven't changed (samples 32 pixels to detect changes)."),
        )
        .arg(
            Arg::with_name("fps")
                .long("fps")
                .takes_value(true)
                .default_value("10")
                .help("Capture framerate in frames per second (default 10)."),
        )
        .get_matches();

    let verbose = matches.is_present("verbose");
    let screenshot = matches.is_present("screenshot");
    let mask_subs = matches.is_present("mask-subtitles");
    let force_hdr_pq = matches.is_present("hdr-pq");
    let no_hdr = matches.is_present("no-hdr");
    let skip_unchanged = matches.is_present("skip-unchanged");

    let lum_hdr: f32 = matches.value_of("luminance-hdr").unwrap().parse().expect("Invalid luminance-hdr value");
    let lum_sdr: f32 = matches.value_of("luminance-sdr").unwrap().parse().expect("Invalid luminance-sdr value");
    let sat_hdr: f32 = matches.value_of("saturation").unwrap().parse().expect("Invalid saturation value");
    let border_frac: f32 = matches.value_of("border").unwrap().parse().expect("Invalid border value");
    let target_fps: f64 = matches.value_of("fps").unwrap().parse().expect("Invalid fps value");
    assert!(
        target_fps.is_finite() && target_fps > 0.0,
        "fps must be a positive number"
    );
    set_luminance_hdr(lum_hdr);
    set_luminance_sdr(lum_sdr);
    set_saturation_hdr(sat_hdr);
    let device_path = matches.value_of("device").unwrap();
    let card = Card::open(device_path);
    let authenticated = card.authenticated().unwrap();

    if verbose {
        let driver = card.get_driver().unwrap();
        println!("Driver (auth={}): {:?}", authenticated, driver);
    }

    unsafe {
        let set_cap = drm_set_client_cap{ capability: drm_ffi::DRM_CLIENT_CAP_UNIVERSAL_PLANES as u64, value: 1 };
        drm_ffi::ioctl::set_cap(card.as_raw_fd(), &set_cap).unwrap();
    }

    // HDR mode override (None = auto-detect per frame)
    let hdr_override: Option<bool> = if force_hdr_pq {
        Some(true)
    } else if no_hdr {
        Some(false)
    } else {
        None
    };

    // Helper to update HDR mode per-frame
    let mut last_hdr_mode: Option<bool> = None;
    let mut last_hdr_check = Instant::now();
    let mut update_hdr_mode = |card: &Card, verbose: bool| {
        let hdr = match hdr_override {
            Some(hdr) => hdr,
            None => {
                // Auto-detection walks every connector and property, which is a
                // burst of ioctls. The mode changes at human timescales, so poll
                // at most once a second instead of on every captured frame.
                if last_hdr_mode.is_none()
                    || last_hdr_check.elapsed() >= Duration::from_secs(1)
                {
                    last_hdr_check = Instant::now();
                    detect_hdr_mode(card, verbose)
                } else {
                    last_hdr_mode.unwrap()
                }
            }
        };
        if last_hdr_mode != Some(hdr) {
            if hdr {
                println!(">>> HDR detected - using PQ tone mapping (luminance={}, saturation={})", lum_hdr, sat_hdr);
            } else {
                println!(">>> SDR mode - using power curve (luminance={})", lum_sdr);
            }
            last_hdr_mode = Some(hdr);
        }
        set_hdr_pq_mode(hdr);
    };

    let adress = matches.value_of("address").unwrap();
    if screenshot {
        if let Some(fb) = find_framebuffer(&card, verbose) {
            update_hdr_mode(&card, verbose);
            let img = dump_framebuffer_to_image(&card, fb, verbose, mask_subs, border_frac).unwrap();
            save_screenshot(&img).unwrap();
        } else {
            println!("No framebuffer found!");
        }
    } else {
        let mut connection = match Connection::connect(adress, verbose, true) {
            Ok(connection) => Some(connection),
            Err(e) => {
                eprintln!(
                    "Could not connect to Hyperion at {} ({}); will keep retrying.",
                    adress, e
                );
                None
            }
        };
        thread::sleep(Duration::from_secs(1));

        let frame_time = Duration::from_secs_f64(1.0 / target_fps);
        let mut prev_samples = [0u32; 32];
        let mut prev_image: Option<RgbImage> = None;
        let mut skipped = 0u32;
        let mut frames = 0u64;
        let mut last_heartbeat = Instant::now();
        let mut last_connect_attempt = Instant::now();
        loop {
            let start = Instant::now();
            // Update HDR mode each frame (auto-detect unless overridden)
            update_hdr_mode(&card, verbose);

            // Build the frame to send: either the previous image (unchanged
            // frame) or a freshly decoded one. `new_image` keeps a decoded frame
            // alive for the duration of the send.
            let mut new_image: Option<RgbImage> = None;
            let image_ref: Option<&RgbImage> = if let Some(fb) = find_framebuffer(&card, verbose) {
                if skip_unchanged {
                    if let Ok(samples) = sample_framebuffer(&card, fb) {
                        if samples == prev_samples {
                            skipped += 1;
                            if verbose {
                                println!("Frame unchanged, resending previous ({})", skipped);
                            }
                            prev_image.as_ref()
                        } else {
                            prev_samples = samples;
                            skipped = 0;
                            new_image = dump_framebuffer_to_image(&card, fb, verbose, mask_subs, border_frac).ok();
                            new_image.as_ref()
                        }
                    } else {
                        None
                    }
                } else {
                    new_image = dump_framebuffer_to_image(&card, fb, verbose, mask_subs, border_frac).ok();
                    new_image.as_ref()
                }
            } else {
                None
            };

            let mut captured = false;
            if let Some(img) = image_ref {
                captured = send_frame(&mut connection, adress, img, verbose, &mut last_connect_attempt);
            }
            if new_image.is_some() {
                prev_image = new_image;
            }
            if captured {
                frames += 1;
            }

            let elapsed = start.elapsed();
            if elapsed < frame_time {
                thread::sleep(frame_time - elapsed);
            }

            // Quiet alive-check: a single line per minute while captures work.
            if last_heartbeat.elapsed() >= Duration::from_secs(60) {
                let window = last_heartbeat.elapsed().as_secs_f64();
                if frames > 0 {
                    println!(
                        "capture OK: {} frames in the last minute ({:.1} fps, {} skipped)",
                        frames,
                        frames as f64 / window,
                        skipped
                    );
                }
                frames = 0;
                last_heartbeat = Instant::now();
            }
        }
    }
}
