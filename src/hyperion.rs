use std::{io::{Cursor, Read, Write}, net::TcpStream};

use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use flatbuffers::FlatBufferBuilder;
use image::{EncodableLayout, RgbImage};

use std::io::Result as StdResult;

use crate::hyperion_reply_generated::hyperionnet as reply;
use crate::hyperion_request_generated::hyperionnet as request;

/// Ask the kernel to ACK the next incoming segment immediately.
///
/// Hyperion frames replies as a 4-byte length followed by a small payload sent
/// as separate writes. If our stack holds the ACK for the length segment
/// (delayed ACK, up to ~40ms) the server's Nagle algorithm cannot release the
/// payload, stalling every reply (two per captured frame here). TCP_QUICKACK
/// makes us ACK promptly so the reply is not held back.
#[cfg(target_os = "linux")]
fn request_quickack(socket: &TcpStream) {
    use std::os::unix::io::AsRawFd;

    let one: libc::c_int = 1;
    unsafe {
        libc::setsockopt(
            socket.as_raw_fd(),
            libc::IPPROTO_TCP,
            libc::TCP_QUICKACK,
            &one as *const _ as *const libc::c_void,
            std::mem::size_of::<libc::c_int>() as libc::socklen_t,
        );
    }
}

#[cfg(not(target_os = "linux"))]
fn request_quickack(_socket: &TcpStream) {}

pub fn read_reply(socket: &mut TcpStream, verbose : bool) -> StdResult<()> {
    request_quickack(socket);
    let mut size = [0u8; 4];
    socket.read_exact(&mut size)?;

    let v = Cursor::new(size).read_u32::<BigEndian>()? as usize;
    let mut msg = vec![0; v];
    socket.read_exact(&mut msg)?;

    let request = reply::root_as_reply(&msg).unwrap();

    if verbose {
        println!("Response {:?}", request);
    }

    Ok(())
}

pub fn register_direct(socket: &mut TcpStream) -> StdResult<()> {
    let mut builder = FlatBufferBuilder::new();

    let origin = builder.create_string("DRM");
    let register = request::Register::create(
        &mut builder,
        &request::RegisterArgs {
            origin: Some(origin),
            priority: 150,
        },
    );
    let offset = request::Request::create(
        &mut builder,
        &request::RequestArgs {
            command_type: request::Command::Register,
            command: Some(register.as_union_value()),
        },
    );
    request::finish_request_buffer(&mut builder, offset);

    let dat = builder.finished_data();

    socket.write_u32::<BigEndian>(dat.len() as _)?;
    socket.write_all(dat)?;
    socket.flush()?;

    Ok(())
}

pub fn send_image(socket: &mut TcpStream, image: & RgbImage, verbose: bool) -> StdResult<()> {
    let mut builder = FlatBufferBuilder::new();

    let raw_bytes = image.as_bytes();

    if verbose {
        println!(
            "Sending image {}x{} (size: {})",
            image.width(),
            image.height(),
            raw_bytes.len()
        );
    }

    let data = builder.create_vector(&raw_bytes);
    let raw_image = request::RawImage::create(
        &mut builder,
        &request::RawImageArgs {
            data: Some(data),
            width: image.width() as _,
            height: image.height() as _,
        },
    );

    let image = request::Image::create(
        &mut builder,
        &request::ImageArgs {
            data_type: request::ImageType::RawImage,
            data: Some(raw_image.as_union_value()),
            duration: 1000,
        },
    );

    let offset = request::Request::create(
        &mut builder,
        &request::RequestArgs {
            command_type: request::Command::Image,
            command: Some(image.as_union_value()),
        },
    );

    request::finish_request_buffer(&mut builder, offset);

    let dat = builder.finished_data();
    socket.write_u32::<BigEndian>(dat.len() as _)?;
    socket.write_all(dat)?;
    socket.flush()?;

    // The reply to an image is not needed to drive the display. Blocking on it
    // serialised every frame behind a network round trip, so it is drained
    // asynchronously instead (see the reader thread in `main`).
    Ok(())
}

pub fn send_color_red(socket: &mut TcpStream, verbose: bool) -> StdResult<()> {
    println!("Setting color");
    let mut builder = flatbuffers::FlatBufferBuilder::new();

    let color = request::Color::create(
        &mut builder,
        &request::ColorArgs {
            data: 0x00100000,
            duration: 5000,
        },
    );

    let offset = request::Request::create(
        &mut builder,
        &request::RequestArgs {
            command_type: request::Command::Color,
            command: Some(color.as_union_value()),
        },
    );

    request::finish_request_buffer(&mut builder, offset);

    let dat = builder.finished_data();
    socket.write_u32::<BigEndian>(dat.len() as _)?;
    socket.write_all(dat)?;
    socket.flush()?;

    read_reply(socket, verbose)?;

    Ok(())
}
