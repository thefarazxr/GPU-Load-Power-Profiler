#include "../include/heavy_hash.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <vector>

__device__ __constant__ uint32_t K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,
    0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
    0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,
    0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,
    0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
    0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,
    0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,
    0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
    0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
};

__device__ inline uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }
__device__ inline uint32_t ch (uint32_t e, uint32_t f, uint32_t g) { return (e & f) ^ (~e & g); }
__device__ inline uint32_t maj(uint32_t a, uint32_t b, uint32_t c) { return (a & b) ^ (a & c) ^ (b & c); }
__device__ inline uint32_t ep0(uint32_t a) { return rotr(a,2)  ^ rotr(a,13) ^ rotr(a,22); }
__device__ inline uint32_t ep1(uint32_t e) { return rotr(e,6)  ^ rotr(e,11) ^ rotr(e,25); }
__device__ inline uint32_t sig0(uint32_t x){ return rotr(x,7)  ^ rotr(x,18) ^ (x >> 3); }
__device__ inline uint32_t sig1(uint32_t x){ return rotr(x,17) ^ rotr(x,19) ^ (x >> 10); }

__device__ void sha256_block(const uint32_t* in, uint32_t* digest) {
    uint32_t w[64];
    #pragma unroll 16
    for (int i = 0; i < 16; ++i) w[i] = in[i];
    #pragma unroll 48
    for (int i = 16; i < 64; ++i)
        w[i] = sig1(w[i-2]) + w[i-7] + sig0(w[i-15]) + w[i-16];

    uint32_t a=digest[0],b=digest[1],c=digest[2],d=digest[3];
    uint32_t e=digest[4],f=digest[5],g=digest[6],h=digest[7];
    #pragma unroll 64
    for (int i = 0; i < 64; ++i) {
        uint32_t t1 = h + ep1(e) + ch(e,f,g) + K[i] + w[i];
        uint32_t t2 = ep0(a) + maj(a,b,c);
        h=g; g=f; f=e; e=d+t1;
        d=c; c=b; b=a; a=t1+t2;
    }
    digest[0]+=a; digest[1]+=b; digest[2]+=c; digest[3]+=d;
    digest[4]+=e; digest[5]+=f; digest[6]+=g; digest[7]+=h;
}

__device__ float fma_chain(const uint32_t* digest, uint32_t rounds) {
    float acc = __uint_as_float((digest[0] & 0x007FFFFFu) | 0x3F800000u);
    float mul = __uint_as_float((digest[1] & 0x007FFFFFu) | 0x3F800000u);
    float add = __uint_as_float((digest[2] & 0x007FFFFFu) | 0x3F000000u);
    #pragma unroll 8
    for (uint32_t i = 0; i < rounds; ++i) {
        acc = __fmaf_rn(acc, mul, add);
        mul = __fmaf_rn(mul, add, acc * 1e-7f);
        add = __fmaf_rn(add, acc, mul * 1e-7f);
    }
    return acc;
}

__device__ uint32_t muladd_mix(uint32_t seed, uint32_t rounds) {
    uint32_t x = seed | 1u;
    #pragma unroll 8
    for (uint32_t i = 0; i < rounds; ++i) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        x  = x * 1664525u + 1013904223u;
        x  = x * 2246822519u ^ (x >> 16);
    }
    return x;
}

__device__ void xorrot_cascade(uint32_t* digest) {
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint32_t j   = (i + 1) & 7;
            uint32_t rot = 7u + (r * 4u) + (uint32_t)i;
            digest[i] ^= rotr(digest[j], rot) + K[i + r*8];
        }
    }
}

// -----------------------------------------------------------------------------
// Strided reads from a large input buffer (buf_elems >> L2 cache capacity)
// create real cache pressure. stride=1 is sequential (L1/L2 friendly);
// stride=32 skips one 128-byte cache line per step; larger strides guarantee
// L2 misses on every iteration. The loaded value is XORed into the SHA-256
// message so the compiler cannot eliminate the read.
// -----------------------------------------------------------------------------
__global__ void heavy_hash_kernel(const uint32_t* __restrict__ input,
                                  uint32_t* __restrict__ output,
                                  uint32_t iters, uint32_t stride,
                                  uint32_t buf_elems) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n   = gridDim.x * blockDim.x;

    uint32_t digest[8] = {
        0x6a09e667u ^ idx, 0xbb67ae85u,
        0x3c6ef372u,       0xa54ff53au ^ idx,
        0x510e527fu,       0x9b05688cu,
        0x1f83d9abu ^ idx, 0x5be0cd19u,
    };

    uint32_t msg[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) msg[i] = idx ^ (uint32_t)(i * 2654435761u);

    for (uint32_t it = 0; it < iters; ++it) {
        // Strided read — cache pressure scales with stride value
        uint32_t mem_val = input[(idx + it * stride) % buf_elems];

        msg[0] = it ^ digest[0] ^ mem_val;  // mix load into message (prevents DCE)

        sha256_block(msg, digest);

        float fp = fma_chain(digest, 64u);
        digest[3] ^= __float_as_uint(fp) & 0x7FFFFFFFu;

        uint32_t mix = muladd_mix(digest[3] ^ digest[7], 64u);
        digest[5] ^= mix;

        xorrot_cascade(digest);

        #pragma unroll
        for (int i = 0; i < 8; ++i) msg[i + 8] = digest[i];
    }

    output[idx % n] = digest[0] ^ digest[7];
}

// ── Host interface ────────────────────────────────────────────────────────────

static void cuda_check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
}

CudaContext heavy_hash_alloc(uint32_t total_threads, uint32_t block_size, uint32_t buf_mb) {
    CudaContext ctx{};
    ctx.total_threads = total_threads;
    ctx.block_size    = block_size;
    ctx.buf_elems     = (buf_mb * 1024u * 1024u) / sizeof(uint32_t);

    cuda_check(cudaMalloc(&ctx.d_input,  ctx.buf_elems    * sizeof(uint32_t)), "alloc input");
    cuda_check(cudaMalloc(&ctx.d_output, ctx.total_threads * sizeof(uint32_t)), "alloc output");

    std::vector<uint32_t> host(ctx.buf_elems);
    for (uint32_t i = 0; i < ctx.buf_elems; ++i) host[i] = i * 2654435761u;
    cuda_check(cudaMemcpy(ctx.d_input, host.data(),
                          ctx.buf_elems * sizeof(uint32_t),
                          cudaMemcpyHostToDevice), "init input");
    return ctx;
}

void heavy_hash_launch(const CudaContext& ctx, uint32_t iters, uint32_t stride) {
    uint32_t grid  = ctx.total_threads / ctx.block_size;
    heavy_hash_kernel<<<grid, ctx.block_size>>>(ctx.d_input, ctx.d_output,
                                                iters, stride, ctx.buf_elems);
    cudaDeviceSynchronize();
    cuda_check(cudaGetLastError(), "kernel");
}

void heavy_hash_free(CudaContext& ctx) {
    cudaFree(ctx.d_input);
    cudaFree(ctx.d_output);
    ctx = {};
}
