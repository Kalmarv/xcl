//! A DXIL module without HLSL: assembled and signed from LLVM IR text (`assemble`).
//!
//! Direct DXIL output (opt-in `XCL_RECOMP_DXIL=1`): the LLVM 3.7 bitcode DXIL is stored as, and the
//! DXBC container with its digest.

pub mod assemble;
pub mod bitstream;
#[cfg(test)]
pub(crate) mod check;
pub mod container;
#[cfg(test)]
pub(crate) mod fixtures;
