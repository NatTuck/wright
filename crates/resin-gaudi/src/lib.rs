//! Host-side step code for the Gaudi model runners: per-step tables written straight into a runner's pinned host
//! buffer, and the step programs that make a step's stream calls. Nothing here links against the Habana
//! libraries, so it builds and tests anywhere.

pub mod card_gate;
pub mod draft_choice;
pub mod ds41;
pub mod fnext;
pub mod glm;
pub mod glue;
pub mod qwen;
pub mod qwen_draft;
pub mod qwen_mtp;
pub mod sampling;
pub mod speculative;
pub mod step;
pub mod tables;
