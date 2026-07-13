#include "ds4_gpu.h"

#include <stdio.h>

int main(void) {
    if (!ds4_gpu_init()) return 1;
    const int ok = ds4_gpu_iq2_q8_tile_selftest();
    ds4_gpu_cleanup();
    if (!ok) return 1;
    puts("cuda IQ2/Q8 tile baseline: OK");
    return 0;
}
