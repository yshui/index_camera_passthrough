Index camera passthrough
========================

**Warning: This is still a work in progress, you could get motion sickness if you try it now**

The problem that the Index camera doesn't work on Linux has been there for a long time, see [ValveSoftware/SteamVR-for-Linux#231](https://github.com/ValveSoftware/SteamVR-for-Linux/issues/231). And Valve doesn't seem to be willing to address it. So I decided to throw something together.

## Current status

For now this application should create an overlay in your game world and show you what your camera sees. This is not proper passthrough yet, i.e. the overlay doesn't move with you, etc.

## TODO

* Make the overlay positioning configurable. e.g. follow controller, distance from your face, etc.
* Implement reprojection. The camera frame rate doesn't match the HMD refresh rate, so we need reprojection to not induce motion sickness
* (Unrealistic) implement Valve's "3D" passthrough. To do this we essentially need to do 3D reconstruction from the stereo camera. There are existing methods, but will be really challenging to implement.

## Build instruction

To build this program, you need:

* Rust ([How to install](https://www.rust-lang.org/tools/install))
* OpenVR
* Vulkan

Make sure you run

```
git submodule update --init
```

in the repository first, then run

```
cargo build --release
```

## Usage

To run this program, you can either

```
cargo run
```

or run the binary directly

```
./target/release/index_camera_passthrough
```

