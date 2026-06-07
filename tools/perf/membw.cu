// tools/perf/membw.cu — synthetic GPU memory-bandwidth probe for GB10.
//
// Measures sustained device-DRAM bandwidth three ways and sweeps buffer size
// so the L2->DRAM knee is visible (don't trust a number taken inside L2):
//   read  : pure streaming read  (most decode-representative; bytes = N)
//   copy  : STREAM copy b[i]=a[i] (bytes = 2N, read+write)
//   triad : a[i]=b[i]+s*c[i]      (bytes = 3N, the classic STREAM number)
//
// Clean-measurement rules baked in: int4 (128-bit) loads, grid-stride with
// enough blocks to saturate, device-resident cudaMalloc (no managed/migratable
// memory), a guarded sink so the read loop can't be DCE'd, warmup + median over
// iters (GB10 DVFS's; median beats max).
//
// Build: nvcc -O3 --use_fast_math -gencode=arch=compute_121a,code=sm_121a \
//          -o /tmp/membw tools/perf/membw.cu
// Run:   /tmp/membw [--max-mb 2048] [--iters 50] [--warmup 10]
//
// GB = 1e9 bytes (decimal, the convention BW vendors quote).

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#define CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(e),__FILE__,__LINE__); \
    exit(1);} } while(0)

__global__ void read_kernel(const int4* __restrict__ in, size_t n4, int4* sink) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    int4 acc = make_int4(0, 0, 0, 0);
    for (; i < n4; i += stride) {
        int4 v = __ldg(&in[i]);
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    // Sentinel the compiler can't prove false -> loads survive DCE.
    if (acc.x == 0x7fffffff && acc.y == acc.z) sink[0] = acc;
}

__global__ void copy_kernel(const int4* __restrict__ in, int4* __restrict__ out, size_t n4) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n4; i += stride) out[i] = __ldg(&in[i]);
}

__global__ void triad_kernel(float* __restrict__ a, const float* __restrict__ b,
                             const float* __restrict__ c, float s, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) a[i] = __ldg(&b[i]) + s * __ldg(&c[i]);
}

static double median(std::vector<float>& v) {
    std::sort(v.begin(), v.end());
    size_t m = v.size() / 2;
    return v.size() & 1 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

// Time a kernel launch `iters` times (after `warmup`), return median GB/s.
template <typename Launch>
static double bench(Launch launch, double bytes, int iters, int warmup) {
    cudaEvent_t a, b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    for (int i = 0; i < warmup; i++) launch();
    CHECK(cudaDeviceSynchronize());
    std::vector<float> ms(iters);
    for (int i = 0; i < iters; i++) {
        CHECK(cudaEventRecord(a));
        launch();
        CHECK(cudaEventRecord(b));
        CHECK(cudaEventSynchronize(b));
        CHECK(cudaEventElapsedTime(&ms[i], a, b));
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    double med_ms = median(ms);
    return bytes / (med_ms * 1e-3) / 1e9;  // GB/s
}

int main(int argc, char** argv) {
    size_t max_mb = 2048; int iters = 50, warmup = 10;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--max-mb") && i + 1 < argc) max_mb = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = atoi(argv[++i]);
    }

    cudaDeviceProp p; CHECK(cudaGetDeviceProperties(&p, 0));
    int sms = p.multiProcessorCount;
    int threads = 256, blocks = sms * 32;  // saturate; grid-stride covers any N
    printf("# membw — %s (sm_%d%d), %d SMs, grid %d x %d threads\n",
           p.name, p.major, p.minor, sms, blocks, threads);
    printf("# read bytes=N  copy bytes=2N  triad bytes=3N   (GB=1e9, median of %d iters)\n", iters);

    size_t max_bytes = max_mb << 20;
    // Three device buffers; triad needs all three. No managed memory.
    int4 *a, *b, *c, *sink;
    CHECK(cudaMalloc(&a, max_bytes));
    CHECK(cudaMalloc(&b, max_bytes));
    CHECK(cudaMalloc(&c, max_bytes));
    CHECK(cudaMalloc(&sink, sizeof(int4)));
    CHECK(cudaMemset(a, 1, max_bytes));
    CHECK(cudaMemset(b, 2, max_bytes));
    CHECK(cudaMemset(c, 3, max_bytes));

    // Sweep: small (in-L2) -> large (DRAM plateau).
    size_t sizes_mb[] = {1, 4, 16, 64, 256, 512, 1024, 2048};
    printf("\n|  size MB | read GB/s | copy GB/s | triad GB/s | %%273 (read) |\n");
    printf("|---------:|----------:|----------:|-----------:|------------:|\n");
    for (size_t smb : sizes_mb) {
        if ((smb << 20) > max_bytes) break;
        size_t bytes = smb << 20;
        size_t n4 = bytes / sizeof(int4);
        size_t nf = bytes / sizeof(float);

        double rd = bench([&]{ read_kernel<<<blocks, threads>>>(a, n4, sink); },
                          (double)bytes, iters, warmup);
        double cp = bench([&]{ copy_kernel<<<blocks, threads>>>(a, b, n4); },
                          2.0 * bytes, iters, warmup);
        double td = bench([&]{ triad_kernel<<<blocks, threads>>>((float*)a, (float*)b,
                          (float*)c, 1.5f, nf); }, 3.0 * bytes, iters, warmup);
        printf("| %8zu | %9.1f | %9.1f | %10.1f | %10.0f%% |\n",
               smb, rd, cp, td, 100.0 * rd / 273.0);
    }
    CHECK(cudaGetLastError());
    cudaFree(a); cudaFree(b); cudaFree(c); cudaFree(sink);
    return 0;
}
