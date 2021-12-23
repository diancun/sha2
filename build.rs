use std::{env, path::PathBuf};

fn main() {
        let library_name = "simdsha2block";
        println!("cargo:rustc-link-lib=static={}", library_name);
        println!("cargo:rustc-link-search=native=/usr/local/lib64/");
}

