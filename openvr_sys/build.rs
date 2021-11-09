fn main() {
    let path = std::path::PathBuf::from("src");
    let mut b = autocxx_build::Builder::new("src/lib.rs", &[&path]).expect_build();
    b.flag_if_supported("-std=c++14").compile("autocxx-demo");

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rustc-link-lib=openvr_api");
}
