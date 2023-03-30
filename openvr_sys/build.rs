fn main() {
    let path = std::path::PathBuf::from("src");
    let b = autocxx_build::Builder::new("src/lib.rs", [&path]);
    b.extra_clang_args(&["-std=c++14"]).build().unwrap().compile("autocxx_demo");

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rustc-link-lib=openvr_api");
}
