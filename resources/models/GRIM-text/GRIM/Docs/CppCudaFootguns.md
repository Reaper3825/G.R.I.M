# C++ / CUDA Footguns

## Vector reference invalidation
Never hold references (`auto& node`) across vector mutations (`emplace_back`, `push_back`). Reallocation invalidates all references.

## C-array initialization
`int arr[256] = {-1};` only sets element 0. Use `std::fill()`.

## Autograd forward `return`
Always explicitly `return output;` from autograd forward functions. See [Autograd.md](Autograd.md).

## Atomic kernel ordering
When kernel B reads atomicAdd output of kernel A, `cudaStreamSynchronize` between them — even on the same stream.

## Gradient norm sync
`computeGradNorm` sync drains the backward pipeline. Pass `sync_for_host_read=false` except when logging.
