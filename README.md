Index camera passthrough
========================

**Warning: This is still a work in progress, you could get motion sickness if you try it now**

The problem that the Index camera doesn't work on Linux has been there for a long time, see [ValveSoftware/SteamVR-for-Linux#231](https://github.com/ValveSoftware/SteamVR-for-Linux/issues/231). And Valve doesn't seem to be willing to address it. So I decided to throw something together.

## Current status

For now this application can create an overlay in your game world that acts as a portal to real world. You can configure the overlay to be in one place, or stay in front of you. This behavior is disabled by default, see [the example config file](index_camera_passthrough.toml) for how to enable and more options.

This program also reads camera calibration data directly from your Steam installation.

## TODO

* Add option to make overlay follow controller.
* Open/close overlay
* (Unrealistic) implement Valve's "3D" passthrough. To do this we essentially need to do 3D reconstruction from the stereo camera. There are existing methods, but will be really challenging to implement.

## Contribute

You can test this out and report your experience to help this improve.

If you have any suggestions about features, or how to make the passthrough look better, please let me know. I am not a graphics programmer and am trying my best to get things work, but solutions I came up with is definitely not going to be as good as things can be.

_Please_ help me out.

## Build instruction

To build this program, you need:

* Rust ([How to install](https://www.rust-lang.org/tools/install), you need to select the nightly channel)
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

### Run from Steam library

After you have built the program, copy it to `/usr/local/bin`

```
cp ./target/release/index_camera_passthrough /usr/local/bin
```

And then add the `index_camera_passthrough.desktop` file to your Steam Library.

### Run directly

To run this program, you can either

```
cargo run
```

or run the binary directly

```
./target/release/index_camera_passthrough
```

