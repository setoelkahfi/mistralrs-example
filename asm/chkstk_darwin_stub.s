// Stub for ___chkstk_darwin — a macOS-only stack probe symbol that does not
// exist on iOS. It is referenced by aws-lc-sys assembly objects even when
// targeting iOS, causing an "undefined symbol" linker error.
//
// On aarch64, ___chkstk_darwin receives the requested stack size in x15
// and is supposed to walk the stack page-by-page to trigger guard pages.
// On iOS the kernel handles stack growth automatically, so a no-op ret
// is sufficient for all practical purposes.

.text
.globl ___chkstk_darwin
.p2align 2
___chkstk_darwin:
    ret
