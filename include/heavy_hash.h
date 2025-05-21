#pragma once
#include <cstdint>

struct CudaContext {
    uint32_t* d_input;
    uint32_t* d_output;
    uint32_t  total_threads;  // grid * block_size
    uint32_t  block_size;     // threads per CUDA block — controls occupancy
    uint32_t  buf_elems;      // input buffer elements (sized to exceed L2 cache)
};

// total_threads = grid_size * block_size; both come from CLI config
CudaContext heavy_hash_alloc(uint32_t total_threads, uint32_t block_size, uint32_t buf_mb);
void        heavy_hash_launch(const CudaContext& ctx, uint32_t iters, uint32_t stride);
void        heavy_hash_free(CudaContext& ctx);
