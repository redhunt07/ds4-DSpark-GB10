// tools/perf/cond_graph_spike.cu — Phase-0 spike for whole-spec-iter device-accept.
//
// Proves the cudaGraphCondTypeIf conditional-node pattern works (and is
// drift-free) on this GB10/sm_121a + driver BEFORE building the real MTP
// device-accept capture. Mirrors glint's spikes/conditional_graph_sm121 and
// ds4-spark's graph-replay-drift findings.
//
// A set_cond kernel reads a device condition and calls cudaGraphSetConditional;
// an IF cond-node's body increments a device counter. We instantiate once and
// replay N times, flipping the condition each replay, and check the body ran
// iff the condition was set — byte-exact across all replays (no executor drift).
//
// Build: /usr/local/cuda/bin/nvcc -O3 -gencode=arch=compute_121a,code=sm_121a \
//          -o /tmp/cond_spike tools/perf/cond_graph_spike.cu
// Run:   /tmp/cond_spike [iters]

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s @ %s:%d\n",cudaGetErrorString(e),__FILE__,__LINE__); return 2;} } while(0)

__global__ void set_cond_kernel(cudaGraphConditionalHandle h, const unsigned *cond) {
    if (threadIdx.x == 0 && blockIdx.x == 0) cudaGraphSetConditional(h, *cond);
}
__global__ void body_kernel(unsigned *body_runs) {
    if (threadIdx.x == 0 && blockIdx.x == 0) atomicAdd(body_runs, 1u);
}

int main(int argc, char **argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 1000;
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("# cond_graph_spike on %s (sm_%d%d), %d replays\n", p.name, p.major, p.minor, iters);
    // Spin scheduler (glint/llama.cpp note: cuts sync latency on sm_121).
    CK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));

    unsigned *cond_dev, *body_runs; unsigned *cond_pinned;
    CK(cudaMalloc(&cond_dev, sizeof(unsigned)));
    CK(cudaMalloc(&body_runs, sizeof(unsigned)));
    CK(cudaMallocHost(&cond_pinned, sizeof(unsigned)));
    CK(cudaMemset(body_runs, 0, sizeof(unsigned)));

    // Build the graph: H2D cond → set_cond (dep H2D) → IF node (dep set_cond);
    // IF body increments body_runs.
    cudaGraph_t g; CK(cudaGraphCreate(&g, 0));
    cudaGraphConditionalHandle handle = 0;
    CK(cudaGraphConditionalHandleCreate(&handle, g, 0, cudaGraphCondAssignDefault));

    cudaGraphNode_t n_h2d = nullptr;
    CK(cudaGraphAddMemcpyNode1D(&n_h2d, g, nullptr, 0, cond_dev, cond_pinned,
                                sizeof(unsigned), cudaMemcpyHostToDevice));
    cudaKernelNodeParams sk = {};
    void *sk_args[] = {&handle, &cond_dev};
    sk.func = (void *)set_cond_kernel; sk.gridDim = dim3(1); sk.blockDim = dim3(1);
    sk.kernelParams = sk_args;
    cudaGraphNode_t n_set = nullptr;
    cudaGraphNode_t h2d_dep[] = {n_h2d};
    CK(cudaGraphAddKernelNode(&n_set, g, h2d_dep, 1, &sk));

    cudaGraphNodeParams ifp = {};
    ifp.type = cudaGraphNodeTypeConditional;
    ifp.conditional.handle = handle;
    ifp.conditional.type = cudaGraphCondTypeIf;
    ifp.conditional.size = 1;
    cudaGraphNode_t n_if = nullptr;
    cudaGraphNode_t set_dep[] = {n_set};
    CK(cudaGraphAddNode(&n_if, g, set_dep, nullptr, 1, &ifp));
    // GOTCHA: phGraph_out is OUTPUT-ONLY — read it AFTER cudaGraphAddNode.
    cudaGraph_t body = ifp.conditional.phGraph_out[0];
    cudaKernelNodeParams bk = {};
    void *bk_args[] = {&body_runs};
    bk.func = (void *)body_kernel; bk.gridDim = dim3(1); bk.blockDim = dim3(1);
    bk.kernelParams = bk_args;
    cudaGraphNode_t n_body = nullptr;
    CK(cudaGraphAddKernelNode(&n_body, body, nullptr, 0, &bk));

    cudaGraphExec_t exec; CK(cudaGraphInstantiate(&exec, g, 0));

    // Replay, flipping the condition; body must run iff cond==1.
    unsigned expected = 0, host_runs = 0;
    int mismatches = 0;
    for (int i = 0; i < iters; i++) {
        *cond_pinned = (i % 3 == 0) ? 1u : 0u;  // arbitrary flip pattern
        if (*cond_pinned) expected++;
        CK(cudaGraphLaunch(exec, 0));
        CK(cudaStreamSynchronize(0));
    }
    CK(cudaMemcpy(&host_runs, body_runs, sizeof(unsigned), cudaMemcpyDeviceToHost));
    if (host_runs != expected) { printf("FAIL: body ran %u, expected %u\n", host_runs, expected); mismatches++; }
    else printf("PASS: cond-node gated body exactly %u/%u replays (drift-free)\n", host_runs, iters);

    cudaGraphExecDestroy(exec); cudaGraphDestroy(g);
    cudaFree(cond_dev); cudaFree(body_runs); cudaFreeHost(cond_pinned);
    return mismatches ? 1 : 0;
}
