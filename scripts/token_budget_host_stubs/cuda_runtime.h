#pragma once
// Host payload tests only need AtomTable's stream type declaration. No CUDA
// function is declared or implemented, so these tests cannot call the runtime.
using cudaStream_t = void*;
