// tools/perf/set_cond_from_accept_spike.cu — Phase-1 spike for whole-spec-iter
// device-accept. Builds the keystone kernel that moves the MTP accept decision
// (currently host-side at ds4.c:19253-19257) onto the device and uses it to
// drive a cudaGraphCondTypeIf branch — so a captured spec-iter can decide
// accept-vs-rollback WITHOUT the per-iter host round-trip
// (ds4_gpu_tensor_read(comp_selected -> row_tops) at ds4.c:14338).
//
// Production accept loop being mirrored (ds4.c:19253-19257):
//     int commit = 1;                               // drafts[0] pre-validated
//     for (int i = 1; i < draft_n; i++) {
//         if (row_tops[i-1] != drafts[i]) break;    // longest matching prefix
//         commit++;
//     }
//
// set_cond_from_accept_kernel computes that same `commit` on-device from the
// device row_tops[]/drafts[] buffers, writes it to a device counter, and calls
// cudaGraphSetConditional(handle, commit == draft_n) — i.e. the IF body (the
// "full-accept tail") runs iff every draft was accepted. Builds directly on the
// validated cond-node from cond_graph_spike.cu (Phase 0).
//
// Differential test: random token patterns; BOTH host and device compute commit
// independently each replay. We check (a) device commit-sum == host commit-sum
// (bit-exact accept logic) and (b) the cond-gated body ran exactly the host
// full-accept count (device commit correctly drives graph control flow), across
// N replays with no executor drift.
//
// Build: /usr/local/cuda/bin/nvcc -O3 -gencode=arch=compute_121a,code=sm_121a \
//          -o /tmp/accept_spike tools/perf/set_cond_from_accept_spike.cu
// Run:   /tmp/accept_spike [iters] [draft_n]

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s @ %s:%d\n",cudaGetErrorString(e),__FILE__,__LINE__); return 2;} } while(0)

#define MAX_DRAFT 8

// Device mirror of the host accept loop. row_tops has draft_n-1 entries
// (verify argmax at positions 0..draft_n-2); drafts has draft_n entries.
__global__ void set_cond_from_accept_kernel(cudaGraphConditionalHandle h,
                                            const int *row_tops, const int *drafts,
                                            int draft_n,
                                            int *commit_out, unsigned *commit_sum) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int commit = 1;                                   // drafts[0] pre-validated
    for (int i = 1; i < draft_n; i++) {
        if (row_tops[i - 1] != drafts[i]) break;      // first mismatch stops accept
        commit++;
    }
    *commit_out = commit;
    atomicAdd(commit_sum, (unsigned)commit);
    cudaGraphSetConditional(h, (commit == draft_n) ? 1u : 0u);  // full-accept gate
}

// Stand-in for the full-accept tail (KEEP_ACCEPTED + snapshot-next-HC) that, in
// the real build, only runs when every draft is accepted.
__global__ void full_accept_tail_kernel(unsigned *full_runs) {
    if (threadIdx.x == 0 && blockIdx.x == 0) atomicAdd(full_runs, 1u);
}

int main(int argc, char **argv) {
    int iters   = argc > 1 ? atoi(argv[1]) : 1000;
    int draft_n = argc > 2 ? atoi(argv[2]) : 3;       // N=3 cascade (default)
    if (draft_n < 2 || draft_n > MAX_DRAFT) { fprintf(stderr, "draft_n in [2,%d]\n", MAX_DRAFT); return 2; }

    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("# set_cond_from_accept_spike on %s (sm_%d%d), %d replays, draft_n=%d\n",
           p.name, p.major, p.minor, iters, draft_n);
    CK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));   // glint/llama.cpp sm_121 sync-latency note

    int *row_tops_dev, *drafts_dev, *commit_out;
    unsigned *commit_sum, *full_runs;
    int *row_tops_pin, *drafts_pin;
    CK(cudaMalloc(&row_tops_dev, MAX_DRAFT * sizeof(int)));
    CK(cudaMalloc(&drafts_dev,   MAX_DRAFT * sizeof(int)));
    CK(cudaMalloc(&commit_out,   sizeof(int)));
    CK(cudaMalloc(&commit_sum,   sizeof(unsigned)));
    CK(cudaMalloc(&full_runs,    sizeof(unsigned)));
    CK(cudaMallocHost(&row_tops_pin, MAX_DRAFT * sizeof(int)));
    CK(cudaMallocHost(&drafts_pin,   MAX_DRAFT * sizeof(int)));
    CK(cudaMemset(commit_sum, 0, sizeof(unsigned)));
    CK(cudaMemset(full_runs,  0, sizeof(unsigned)));

    // Graph: H2D row_tops + H2D drafts -> set_cond_from_accept (dep both) ->
    // IF(full-accept) { full_accept_tail }.
    cudaGraph_t g; CK(cudaGraphCreate(&g, 0));
    cudaGraphConditionalHandle handle = 0;
    CK(cudaGraphConditionalHandleCreate(&handle, g, 0, cudaGraphCondAssignDefault));

    cudaGraphNode_t n_h2d_rt = nullptr, n_h2d_dr = nullptr;
    CK(cudaGraphAddMemcpyNode1D(&n_h2d_rt, g, nullptr, 0, row_tops_dev, row_tops_pin,
                                MAX_DRAFT * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaGraphAddMemcpyNode1D(&n_h2d_dr, g, nullptr, 0, drafts_dev, drafts_pin,
                                MAX_DRAFT * sizeof(int), cudaMemcpyHostToDevice));

    cudaKernelNodeParams sk = {};
    void *sk_args[] = {&handle, &row_tops_dev, &drafts_dev, &draft_n, &commit_out, &commit_sum};
    sk.func = (void *)set_cond_from_accept_kernel; sk.gridDim = dim3(1); sk.blockDim = dim3(1);
    sk.kernelParams = sk_args;
    cudaGraphNode_t n_set = nullptr;
    cudaGraphNode_t set_deps[] = {n_h2d_rt, n_h2d_dr};
    CK(cudaGraphAddKernelNode(&n_set, g, set_deps, 2, &sk));

    cudaGraphNodeParams ifp = {};
    ifp.type = cudaGraphNodeTypeConditional;
    ifp.conditional.handle = handle;
    ifp.conditional.type = cudaGraphCondTypeIf;
    ifp.conditional.size = 1;
    cudaGraphNode_t n_if = nullptr;
    cudaGraphNode_t if_dep[] = {n_set};
    CK(cudaGraphAddNode(&n_if, g, if_dep, nullptr, 1, &ifp));
    // GOTCHA: phGraph_out is OUTPUT-ONLY — read it AFTER cudaGraphAddNode.
    cudaGraph_t body = ifp.conditional.phGraph_out[0];
    cudaKernelNodeParams bk = {};
    void *bk_args[] = {&full_runs};
    bk.func = (void *)full_accept_tail_kernel; bk.gridDim = dim3(1); bk.blockDim = dim3(1);
    bk.kernelParams = bk_args;
    cudaGraphNode_t n_body = nullptr;
    CK(cudaGraphAddKernelNode(&n_body, body, nullptr, 0, &bk));

    cudaGraphExec_t exec; CK(cudaGraphInstantiate(&exec, g, 0));

    // Differential replay: random token patterns, host + device each compute commit.
    unsigned host_commit_sum = 0, host_full_runs = 0;
    unsigned long lcg = 0x9e3779b97f4a7c15ULL;  // deterministic
    int cov[MAX_DRAFT + 1] = {0};                // commit-value coverage histogram
    for (int it = 0; it < iters; it++) {
        // Fill drafts[] and row_tops[] with small token ids; bias toward matches
        // so all commit values 1..draft_n get exercised.
        for (int i = 0; i < draft_n; i++) {
            lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
            drafts_pin[i] = (int)((lcg >> 33) % 5);          // tokens in [0,5)
        }
        for (int i = 0; i < draft_n - 1; i++) {
            lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
            // 60% chance to match drafts[i+1] (so prefixes of varying length form)
            int match = ((lcg >> 33) % 5) < 3;
            row_tops_pin[i] = match ? drafts_pin[i + 1] : (int)(((lcg >> 20) % 5) + 5);
        }
        // Host reference accept loop (the spec being mirrored).
        int hcommit = 1;
        for (int i = 1; i < draft_n; i++) { if (row_tops_pin[i - 1] != drafts_pin[i]) break; hcommit++; }
        host_commit_sum += (unsigned)hcommit;
        if (hcommit == draft_n) host_full_runs++;
        cov[hcommit]++;

        CK(cudaGraphLaunch(exec, 0));
        CK(cudaStreamSynchronize(0));
    }

    unsigned dev_commit_sum = 0, dev_full_runs = 0;
    CK(cudaMemcpy(&dev_commit_sum, commit_sum, sizeof(unsigned), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&dev_full_runs,  full_runs,  sizeof(unsigned), cudaMemcpyDeviceToHost));

    printf("# commit coverage:");
    for (int c = 1; c <= draft_n; c++) printf(" commit=%d:%d", c, cov[c]);
    printf("\n");

    int fail = 0;
    if (dev_commit_sum != host_commit_sum) {
        printf("FAIL: device commit-sum %u != host %u (accept logic mismatch)\n", dev_commit_sum, host_commit_sum);
        fail = 1;
    } else {
        printf("PASS: device commit-sum == host %u (accept logic bit-exact over %d replays)\n", host_commit_sum, iters);
    }
    if (dev_full_runs != host_full_runs) {
        printf("FAIL: cond body ran %u, expected %u (device commit drove branch wrong)\n", dev_full_runs, host_full_runs);
        fail = 1;
    } else {
        printf("PASS: full-accept tail gated exactly %u/%d replays (device commit drives cond-node, drift-free)\n",
               host_full_runs, iters);
    }

    cudaGraphExecDestroy(exec); cudaGraphDestroy(g);
    cudaFree(row_tops_dev); cudaFree(drafts_dev); cudaFree(commit_out);
    cudaFree(commit_sum); cudaFree(full_runs);
    cudaFreeHost(row_tops_pin); cudaFreeHost(drafts_pin);
    return fail ? 1 : 0;
}
