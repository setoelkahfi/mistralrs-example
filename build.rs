use std::env;

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    if target_os == "ios" {
        // Compile the ___chkstk_darwin stub for iOS.
        // aws-lc-sys references this macOS-only symbol even when targeting iOS,
        // causing an undefined symbol linker error. We provide a no-op stub.
        cc::Build::new()
            .file("asm/chkstk_darwin_stub.s")
            .compile("chkstk_darwin_stub");
    }
}
