#include "ds4.h"
#include "ds4_distributed.h"
#include "ds4_help.h"
#include "ds4_kvstore.h"

/* Purpose-built throughput benchmark.
 *
 * The benchmark walks one fixed token sequence to configurable context
 * frontiers, measuring only the newest prefill interval at each frontier.  It
 * then snapshots the live session in memory, performs a fixed greedy decode
 * run without allowing EOS, restores the snapshot, and continues to the next
 * frontier.  Snapshot save/restore time is intentionally outside both timing
 * windows.
 */

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    const char *model_path;
    const char *prompt_path;
    const char *chat_prompt_path;
    const char *kv_restore_path;
    const char *system;
    const char *csv_path;
    const char *expert_profile_path;
    ds4_backend backend;
    int threads;
    int ctx_start;
    int ctx_max;
    int ctx_alloc;
    int step_incr;
    int gen_tokens;
    int power_percent;
    uint32_t prefill_chunk;
    uint32_t ssd_streaming_cache_experts;
    uint64_t ssd_streaming_cache_bytes;
    uint32_t ssd_streaming_preload_experts;
    uint64_t simulate_used_memory_bytes;
    double step_mul;
    const char *dump_frontier_logits_dir;
    ds4_dist_options dist;
    const char *mtp_path;
    int mtp_draft_tokens;
    bool warm_weights;
    bool quality;
    bool ssd_streaming;
    bool ssd_streaming_cold;
    float temperature;   /* >0 => sampled decode (spec-sampling when --mtp); 0 => greedy */
    float top_p;
    float min_p;
    uint64_t seed;
    bool batch_cost;     /* multi-stream batch cost curve instead of the gen loop */
    int batch_cost_w;    /* restrict --batch-cost to one width (nsys attribution) */
    int batch_check;     /* token-exactness gate: plain vs batch replay over N tokens */
} bench_config;

static double bench_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static void usage(FILE *fp, const char *topic) {
    ds4_help_print(fp, DS4_HELP_BENCH, topic);
}

static int parse_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v <= 0 || v > INT_MAX) {
        fprintf(stderr, "ds4-bench: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static int parse_nonnegative_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v < 0 || v > INT_MAX) {
        fprintf(stderr, "ds4-bench: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static double parse_double_arg(const char *s, const char *opt) {
    char *end = NULL;
    double v = strtod(s, &end);
    if (s[0] == '\0' || *end != '\0' || !isfinite(v)) {
        fprintf(stderr, "ds4-bench: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return v;
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4-bench: %s requires an argument\n", opt);
        exit(2);
    }
    return argv[++*i];
}

static ds4_backend parse_backend(const char *s, const char *opt) {
    if (!strcmp(s, "metal")) return DS4_BACKEND_METAL;
#ifdef DS4_ROCM_BUILD
    if (!strcmp(s, "rocm")) return DS4_BACKEND_CUDA;
#else
    if (!strcmp(s, "cuda")) return DS4_BACKEND_CUDA;
#endif
    if (!strcmp(s, "cpu")) return DS4_BACKEND_CPU;
    fprintf(stderr, "ds4-bench: invalid value for %s: %s\n", opt, s);
#ifdef DS4_ROCM_BUILD
    fprintf(stderr, "ds4-bench: valid backends are: metal, rocm, cpu\n");
#else
    fprintf(stderr, "ds4-bench: valid backends are: metal, cuda, cpu\n");
#endif
    exit(2);
}

static ds4_backend default_backend(void) {
#ifdef DS4_NO_GPU
    return DS4_BACKEND_CPU;
#elif defined(__APPLE__)
    return DS4_BACKEND_METAL;
#else
    return DS4_BACKEND_CUDA;
#endif
}

static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ds4-bench: failed to open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "ds4-bench: failed to seek %s\n", path);
        fclose(fp);
        exit(1);
    }
    long n = ftell(fp);
    if (n < 0) {
        fprintf(stderr, "ds4-bench: failed to tell %s\n", path);
        fclose(fp);
        exit(1);
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        fprintf(stderr, "ds4-bench: failed to rewind %s\n", path);
        fclose(fp);
        exit(1);
    }
    char *buf = malloc((size_t)n + 1);
    if (!buf) {
        fprintf(stderr, "ds4-bench: out of memory reading %s\n", path);
        fclose(fp);
        exit(1);
    }
    if (fread(buf, 1, (size_t)n, fp) != (size_t)n) {
        fprintf(stderr, "ds4-bench: failed to read %s\n", path);
        free(buf);
        fclose(fp);
        exit(1);
    }
    fclose(fp);
    buf[n] = '\0';
    return buf;
}

static bench_config parse_options(int argc, char **argv) {
    bench_config c = {
        .model_path = "ds4flash.gguf",
        .system = "You are a helpful assistant.",
        .backend = default_backend(),
        .ctx_start = 2048,
        .ctx_max = 32768,
        .step_incr = 2048,
        .gen_tokens = 128,
        .step_mul = 1.0,
        .mtp_draft_tokens = 2,
        .temperature = 0.0f,        /* greedy by default */
        .top_p = 0.95f,
        .min_p = 0.0f,
        .seed = 1234,
    };

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            const char *topic = (i + 1 < argc && argv[i + 1][0] != '-') ?
                argv[i + 1] : NULL;
            usage(stdout, topic);
            exit(0);
        }
        char dist_parse_err[256] = {0};
        ds4_dist_cli_parse_result dist_parse =
            ds4_dist_parse_cli_arg(arg,
                                   &i,
                                   argc,
                                   argv,
                                   &c.dist,
                                   dist_parse_err,
                                   sizeof(dist_parse_err));
        if (dist_parse == DS4_DIST_CLI_ERROR) {
            fprintf(stderr,
                    "ds4-bench: %s\n",
                    dist_parse_err[0] ? dist_parse_err : "invalid distributed option");
            exit(2);
        }
        if (dist_parse == DS4_DIST_CLI_MATCHED) continue;

        if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            c.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--prompt-file")) {
            c.prompt_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--chat-prompt-file")) {
            c.chat_prompt_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--kv-restore")) {
            c.kv_restore_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-sys") || !strcmp(arg, "--system")) {
            c.system = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--ctx-start")) {
            c.ctx_start = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--ctx-max")) {
            c.ctx_max = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--ctx-alloc")) {
            c.ctx_alloc = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--step-incr")) {
            c.step_incr = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--step-mul")) {
            c.step_mul = parse_double_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--gen-tokens") || !strcmp(arg, "--tokens") || !strcmp(arg, "-n")) {
            c.gen_tokens = parse_nonnegative_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--csv")) {
            c.csv_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--batch-cost")) {
            c.batch_cost = true;
        } else if (!strcmp(arg, "--batch-cost-w")) {
            c.batch_cost = true;
            c.batch_cost_w = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--batch-check")) {
            c.batch_cost = true;   /* same dispatch + validation as --batch-cost */
            c.batch_check = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--dump-frontier-logits-dir")) {
            c.dump_frontier_logits_dir = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--expert-profile")) {
            c.expert_profile_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) {
            c.threads = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--backend")) {
            c.backend = parse_backend(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--metal")) {
            c.backend = DS4_BACKEND_METAL;
#ifdef DS4_ROCM_BUILD
        } else if (!strcmp(arg, "--rocm")) {
            c.backend = DS4_BACKEND_CUDA;
#else
        } else if (!strcmp(arg, "--cuda")) {
            c.backend = DS4_BACKEND_CUDA;
#endif
        } else if (!strcmp(arg, "--cpu")) {
            c.backend = DS4_BACKEND_CPU;
        } else if (!strcmp(arg, "--quality")) {
            c.quality = true;
        } else if (!strcmp(arg, "--ssd-streaming")) {
            c.ssd_streaming = true;
        } else if (!strcmp(arg, "--ssd-streaming-cold")) {
            c.ssd_streaming_cold = true;
        } else if (!strcmp(arg, "--ssd-streaming-cache-experts")) {
            uint32_t experts = 0;
            uint64_t bytes = 0;
            if (!ds4_parse_streaming_cache_experts_arg(
                    need_arg(&i, argc, argv, arg), &experts, &bytes)) {
                fprintf(stderr,
                        "ds4-bench: --ssd-streaming-cache-experts must be a positive count or <number>GB\n");
                exit(2);
            }
            c.ssd_streaming_cache_experts = experts;
            c.ssd_streaming_cache_bytes = bytes;
        } else if (!strcmp(arg, "--ssd-streaming-preload-experts")) {
            int v = parse_int(need_arg(&i, argc, argv, arg), arg);
            if (v <= 0) {
                fprintf(stderr, "ds4-bench: --ssd-streaming-preload-experts must be positive\n");
                exit(2);
            }
            c.ssd_streaming_preload_experts = (uint32_t)v;
        } else if (!strcmp(arg, "--simulate-used-memory")) {
            if (!ds4_parse_gib_arg(need_arg(&i, argc, argv, arg),
                                   &c.simulate_used_memory_bytes)) {
                fprintf(stderr,
                        "ds4-bench: --simulate-used-memory must be a positive GiB value, e.g. 64GB\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--prefill-chunk")) {
            c.prefill_chunk = (uint32_t)parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--power")) {
            c.power_percent = parse_int(need_arg(&i, argc, argv, arg), arg);
            if (c.power_percent < 1 || c.power_percent > 100) {
                fprintf(stderr, "ds4-bench: --power must be between 1 and 100\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--warm-weights")) {
            c.warm_weights = true;
        } else if (!strcmp(arg, "--mtp")) {
            c.mtp_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--mtp-draft")) {
            c.mtp_draft_tokens = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--temp")) {
            c.temperature = (float)parse_double_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--top-p")) {
            c.top_p = (float)parse_double_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--min-p")) {
            c.min_p = (float)parse_double_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--seed")) {
            c.seed = (uint64_t)strtoull(need_arg(&i, argc, argv, arg), NULL, 10);
        } else {
            fprintf(stderr, "ds4-bench: unknown option: %s\n", arg);
            usage(stderr, NULL);
            exit(2);
        }
    }

    if (c.kv_restore_path) {
        if (c.prompt_path || c.chat_prompt_path) {
            fprintf(stderr, "ds4-bench: --kv-restore is mutually exclusive with --prompt-file/--chat-prompt-file\n");
            exit(2);
        }
    } else if (!!c.prompt_path == !!c.chat_prompt_path) {
        fprintf(stderr, "ds4-bench: specify exactly one of --prompt-file or --chat-prompt-file\n");
        exit(2);
    }
    if (c.ctx_start > c.ctx_max) {
        fprintf(stderr, "ds4-bench: --ctx-start must be <= --ctx-max\n");
        exit(2);
    }
    if (c.step_mul < 1.0) {
        fprintf(stderr, "ds4-bench: --step-mul must be >= 1\n");
        exit(2);
    }
    if (c.step_mul == 1.0 && c.step_incr <= 0) {
        fprintf(stderr, "ds4-bench: --step-incr must be positive when --step-mul is 1\n");
        exit(2);
    }
    if (c.batch_cost) {
        if (!c.mtp_path) {
            fprintf(stderr, "ds4-bench: --batch-cost needs --mtp (verify scratch buffers; "
                            "the draft head is not called)\n");
            exit(2);
        }
        /* The replay teacher-forces up to (warmup+rounds)*Σ(1..5) = 270 prompt
         * tokens past each frontier; gen_tokens only sizes ctx_alloc here (the
         * gen loop is bypassed), so raise it to cover the replay headroom. */
        if (c.gen_tokens < 280) c.gen_tokens = 280;
    }
    if (c.ctx_max > INT_MAX - c.gen_tokens - 1) {
        fprintf(stderr, "ds4-bench: requested context is too large\n");
        exit(2);
    }
    if (c.ctx_alloc == 0) c.ctx_alloc = c.ctx_max + c.gen_tokens + 1;
    if (c.ctx_alloc <= c.ctx_max + c.gen_tokens) {
        fprintf(stderr, "ds4-bench: --ctx-alloc must be greater than ctx-max + gen-tokens\n");
        exit(2);
    }
    char dist_err[256];
    if (ds4_dist_prepare_engine_options(&c.dist, NULL, dist_err, sizeof(dist_err)) != 0) {
        fprintf(stderr, "ds4-bench: %s\n", dist_err);
        exit(2);
    }
    if (c.dist.role == DS4_DISTRIBUTED_WORKER) {
        fprintf(stderr, "ds4-bench: --role worker is a serving mode; start workers with ./ds4\n");
        exit(2);
    }
    return c;
}

static void json_write_string(FILE *fp, const char *s) {
    fputc('"', fp);
    if (s) {
        for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
            switch (*p) {
            case '"':  fputs("\\\"", fp); break;
            case '\\': fputs("\\\\", fp); break;
            case '\b': fputs("\\b", fp); break;
            case '\f': fputs("\\f", fp); break;
            case '\n': fputs("\\n", fp); break;
            case '\r': fputs("\\r", fp); break;
            case '\t': fputs("\\t", fp); break;
            default:
                if (*p < 0x20) fprintf(fp, "\\u%04x", (unsigned)*p);
                else fputc((char)*p, fp);
                break;
            }
        }
    }
    fputc('"', fp);
}

static int write_frontier_logits_json(
        const bench_config *cfg,
        ds4_engine         *engine,
        ds4_session        *session,
        int                 frontier,
        int                 previous) {
    if (!cfg->dump_frontier_logits_dir) return 0;

    const int vocab = ds4_engine_vocab_size(engine);
    float *logits = malloc((size_t)vocab * sizeof(logits[0]));
    if (!logits) {
        fprintf(stderr, "ds4-bench: out of memory copying frontier logits\n");
        return 1;
    }
    if (ds4_session_copy_logits(session, logits, vocab) != vocab) {
        fprintf(stderr, "ds4-bench: failed to copy frontier logits at %d\n", frontier);
        free(logits);
        return 1;
    }

    char path[PATH_MAX];
    const int n = snprintf(path,
                           sizeof(path),
                           "%s/frontier_%06d.logits.json",
                           cfg->dump_frontier_logits_dir,
                           frontier);
    if (n <= 0 || (size_t)n >= sizeof(path)) {
        fprintf(stderr, "ds4-bench: frontier logits path is too long\n");
        free(logits);
        return 1;
    }

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4-bench: failed to open %s: %s\n", path, strerror(errno));
        free(logits);
        return 1;
    }

    const int argmax = ds4_session_argmax(session);
    fprintf(fp, "{\n  \"source\":\"ds4-bench\",\n  \"model\":");
    json_write_string(fp, cfg->model_path);
    fprintf(fp,
            ",\n  \"backend\":\"%s\",\n  \"quality\":%s,\n"
            "  \"quant_bits\":%d,\n  \"prompt_tokens\":%d,\n"
            "  \"frontier_tokens\":%d,\n  \"prefill_tokens\":%d,\n"
            "  \"ctx\":%d,\n  \"vocab\":%d,\n"
            "  \"argmax_id\":%d,\n  \"argmax_logit\":%.9g,\n  \"logits\":[",
            ds4_backend_name(cfg->backend),
            cfg->quality ? "true" : "false",
            ds4_engine_routed_quant_bits(engine),
            frontier,
            frontier,
            frontier - previous,
            cfg->ctx_alloc,
            vocab,
            argmax,
            logits[argmax]);
    for (int i = 0; i < vocab; i++) {
        if (i) fputc(',', fp);
        if ((i % 8) == 0) fputs("\n    ", fp);
        if (isfinite(logits[i])) fprintf(fp, "%.9g", logits[i]);
        else fputs("null", fp);
    }
    fputs("\n  ]\n}\n", fp);
    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4-bench: failed to close %s\n", path);
        free(logits);
        return 1;
    }
    free(logits);
    return 0;
}

static int next_frontier(const bench_config *c, int cur) {
    if (cur >= c->ctx_max) return c->ctx_max;
    int next;
    if (c->step_mul == 1.0) {
        if (cur > INT_MAX - c->step_incr) next = c->ctx_max;
        else next = cur + c->step_incr;
    } else {
        const double v = ceil((double)cur * c->step_mul);
        next = v > (double)INT_MAX ? c->ctx_max : (int)v;
        if (next <= cur) next = cur + 1;
    }
    if (next > c->ctx_max) next = c->ctx_max;
    return next;
}

static void log_context_memory(ds4_backend backend,
                               int         ctx_size,
                               uint32_t    prefill_chunk) {
    ds4_context_memory m =
        ds4_context_memory_estimate_with_prefill(backend,
                                                 ctx_size,
                                                 prefill_chunk);
    fprintf(stderr,
            "ds4-bench: context buffers %.2f MiB (ctx=%d, backend=%s, prefill_chunk=%u, raw_kv_rows=%u, compressed_kv_rows=%u)\n",
            (double)m.total_bytes / (1024.0 * 1024.0),
            ctx_size,
            ds4_backend_name(backend),
            m.prefill_cap,
            m.raw_cap,
            m.comp_cap);
}

static int wait_distributed_route(ds4_session *session) {
    char err[256] = {0};
    char last[256] = {0};
    unsigned ticks = 0;
    const struct timespec delay = {0, 250000000L};

    for (;;) {
        int ready = ds4_session_distributed_route_ready(session, err, sizeof(err));
        if (ready > 0) {
            if (ticks) fprintf(stderr, "ds4-bench: distributed route ready\n");
            return 0;
        }
        if (ready < 0) {
            fprintf(stderr,
                    "ds4-bench: distributed route readiness failed: %s\n",
                    err[0] ? err : "unknown error");
            return 1;
        }
        const char *why = err[0] ? err : "route incomplete";
        if (strcmp(last, why) != 0 || (ticks % 20u) == 0) {
            fprintf(stderr, "ds4-bench: waiting for distributed route: %s\n", why);
            snprintf(last, sizeof(last), "%s", why);
        }
        nanosleep(&delay, NULL);
        ticks++;
    }
}

static void maybe_warn_distributed_step_shape(const bench_config *cfg, ds4_session *session) {
    if (!cfg || !session || cfg->dist.role != DS4_DISTRIBUTED_COORDINATOR) return;
    uint32_t chunk = cfg->dist.prefill_chunk;
    if (chunk == 0) {
        const int cap = ds4_session_prefill_cap(session);
        if (cap > 0) chunk = (uint32_t)cap;
    }
    if (chunk == 0) return;
    if (cfg->step_mul == 1.0 &&
        cfg->step_incr > 0 &&
        (uint32_t)cfg->step_incr < chunk &&
        cfg->ctx_start < cfg->ctx_max)
    {
        fprintf(stderr,
                "ds4-bench: note: --step-incr=%d is smaller than distributed prefill chunk %u; "
                "suffix rows will not show multi-chunk pipeline overlap\n",
                cfg->step_incr,
                chunk);
    }
}

/* Multi-stream batched-decode cost curve (--batch-cost).
 *
 * Plain decode on GB10 is membw-bound per STREAM: one decode step reads the
 * full hot weight set whether it serves 1 row or 5.  This measures what one
 * fused forward costs at width w = 1..5 rows (the per-row KV/position
 * indirection of true multi-session batching is NOT included — rows here are
 * consecutive teacher-forced positions of one stream, so the numbers are the
 * OPTIMISTIC bound that decides go/no-go on building real multi-session
 * batching).  Widths are interleaved round-robin so depth drift averages out
 * across widths instead of biasing the wide rows deeper.
 *
 * CSV rows: ctx_tokens,width,steps,ms_mean,ms_std,agg_tps,agg_speedup_vs_w1 */
/* Token-exactness gate for the batch replay (--batch-check N).
 *
 * Pass A teacher-forces N prompt tokens through the plain decode path
 * (eval_token_raw_swa), recording the greedy argmax after every position.
 * Pass B replays the SAME positions through the batched verify forward at
 * widths 2..5 (row_tops give rows 0..w-2; the wrapper's last-row logits give
 * row w-1) and compares argmaxes position-by-position.  Combined batch-N MoE /
 * attention is documented as not bit-identical to N=1 raw_swa (near-tie
 * drift), so the gate reports the agreement RATE per width, not a hard
 * assert.  MoE down atomics are forced off (same rule as token-diff). */
static int run_batch_check(ds4_session *session, ds4_engine *engine,
                           const ds4_tokens *prompt,
                           int frontier, int n_check) {
    char err[256];
    if (n_check > prompt->len - frontier) n_check = prompt->len - frontier;
    n_check -= n_check % 60;   /* divisible by all widths 2..5 (and 60 = lcm) */
    if (n_check <= 0) {
        fprintf(stderr, "ds4-bench: batch-check needs >= 60 prompt tokens past the frontier\n");
        return 1;
    }
    /* Same determinism rule as token-diff: atomic MoE down flips near-tie
     * argmaxes run-to-run and would pollute the agreement rate. */
    if (!getenv("DS4_CUDA_MOE_NO_ATOMIC_DOWN")) {
        setenv("DS4_CUDA_MOE_NO_ATOMIC_DOWN", "1", 0);
        fprintf(stderr, "ds4-bench: DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 auto-set (batch-check)\n");
    }

    /* Full-logit reference for the first LOGIT_N positions (~32 MB): the
     * width-1 pass below compares the complete vocab vector per position
     * (max-abs + rms), a much stronger fidelity gate than argmax identity. */
    enum { LOGIT_N = 64 };
    const int n_vocab = ds4_engine_vocab_size(engine);
    const int n_logit = n_check < LOGIT_N ? n_check : LOGIT_N;
    float *ref_logits = malloc((size_t)n_logit * (size_t)n_vocab * sizeof(float));
    int *plain_top = malloc((size_t)n_check * sizeof(int));
    int *batch_top = malloc((size_t)n_check * sizeof(int));
    if (!plain_top || !batch_top || !ref_logits) {
        free(plain_top); free(batch_top); free(ref_logits);
        return 1;
    }

    ds4_session_snapshot snap = {0};
    int rc = 1;
    if (ds4_session_save_snapshot(session, &snap, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4-bench: batch-check snapshot failed: %s\n", err);
        goto done;
    }

    for (int i = 0; i < n_check; i++) {
        if (ds4_session_eval(session, prompt->v[frontier + i], err, sizeof(err)) != 0) {
            fprintf(stderr, "ds4-bench: batch-check plain eval at +%d failed: %s\n", i, err);
            goto done;
        }
        plain_top[i] = ds4_session_argmax(session);
        if (i < n_logit)
            ds4_session_copy_logits(session, ref_logits + (size_t)i * n_vocab, n_vocab);
    }

    printf("batch-check: ctx=%d n=%d (plain argmax reference captured)\n", frontier, n_check);

    /* Width-1 pass: replay every position individually through the batch
     * forward and compare the FULL logit vector against the plain reference. */
    if (ds4_session_load_snapshot(session, &snap, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4-bench: batch-check restore failed: %s\n", err);
        goto done;
    }
    {
        double worst_abs = 0.0, worst_rms = 0.0;
        int worst_pos = -1, w1_mis = 0;
        float *cur = malloc((size_t)n_vocab * sizeof(float));
        if (!cur) goto done;
        for (int i = 0; i < n_logit; i++) {
            if (ds4_session_eval_batch_replay(session, prompt->v + frontier + i, 1,
                                              err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-bench: batch-check replay w=1 at +%d failed: %s\n",
                        i, err);
                free(cur);
                goto done;
            }
            if (ds4_session_argmax(session) != plain_top[i]) w1_mis++;
            ds4_session_copy_logits(session, cur, n_vocab);
            const float *ref = ref_logits + (size_t)i * n_vocab;
            double sumsq = 0.0, mx = 0.0;
            for (int v = 0; v < n_vocab; v++) {
                const double d = fabs((double)cur[v] - (double)ref[v]);
                if (d > mx) mx = d;
                sumsq += d * d;
            }
            const double rms = sqrt(sumsq / n_vocab);
            if (mx > worst_abs) { worst_abs = mx; worst_pos = i; }
            if (rms > worst_rms) worst_rms = rms;
        }
        free(cur);
        printf("batch-check: w=1 logits over %d positions: worst_maxabs=%.6f (at +%d) "
               "worst_rms=%.6f argmax_mismatches=%d\n",
               n_logit, worst_abs, worst_pos, worst_rms, w1_mis);
    }
    for (int w = 2; w <= 5; w++) {
        if (ds4_session_load_snapshot(session, &snap, err, sizeof(err)) != 0) {
            fprintf(stderr, "ds4-bench: batch-check restore failed: %s\n", err);
            goto done;
        }
        for (int pos = 0; pos < n_check; pos += w) {
            if (ds4_session_eval_batch_replay(session, prompt->v + frontier + pos, w,
                                              err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-bench: batch-check replay w=%d at +%d failed: %s\n",
                        w, pos, err);
                goto done;
            }
            /* The wrapper exposes only the LAST row's logits, so each width w
             * samples positions ≡ w-1 (mod w).  Across w=2..5 that covers most
             * positions; the agreement rate over the sampled subset is the
             * gate, not exhaustive per-position equality. */
            batch_top[pos + w - 1] = ds4_session_argmax(session);
        }
        int n_cmp = 0, n_mis = 0, first_mis = -1;
        for (int i = w - 1; i < n_check; i += w) {
            n_cmp++;
            if (batch_top[i] != plain_top[i]) {
                n_mis++;
                if (first_mis < 0) first_mis = i;
            }
        }
        if (first_mis < 0) {
            printf("batch-check: w=%d compared=%d mismatches=0 (100.00%% agree)\n",
                   w, n_cmp);
        } else {
            printf("batch-check: w=%d compared=%d mismatches=%d (%.2f%% agree) "
                   "first_mismatch=+%d\n",
                   w, n_cmp, n_mis, 100.0 * (n_cmp - n_mis) / n_cmp, first_mis);
        }
    }
    rc = 0;
done:
    ds4_session_snapshot_free(&snap);
    free(ref_logits);
    free(plain_top);
    free(batch_top);
    return rc;
}

static int run_batch_cost(ds4_session *session, const ds4_tokens *prompt,
                          int frontier, FILE *out, int only_w) {
    enum { WMAX = 5, ROUNDS = 16, WARMUP = 2 };
    const int per_round = WMAX * (WMAX + 1) / 2;   /* sum of widths 1..WMAX */
    char err[256];

    const int w_lo = only_w > 0 ? only_w : 1;
    const int w_hi = only_w > 0 ? only_w : WMAX;
    const int round_toks = only_w > 0 ? only_w : per_round;

    const int avail = prompt->len - frontier;
    int rounds = ROUNDS;
    while (rounds > 1 && (WARMUP + rounds) * round_toks > avail) rounds--;
    if ((WARMUP + rounds) * round_toks > avail) {
        fprintf(stderr, "ds4-bench: batch-cost needs >= %d prompt tokens past the "
                        "frontier (%d available)\n",
                (WARMUP + 1) * round_toks, avail);
        return 1;
    }
    if (rounds < ROUNDS)
        fprintf(stderr, "ds4-bench: batch-cost rounds trimmed to %d (short prompt)\n",
                rounds);

    double sum[WMAX + 1] = {0}, sumsq[WMAX + 1] = {0};
    int cnt[WMAX + 1] = {0};
    int pos = frontier;
    for (int r = 0; r < WARMUP + rounds; r++) {
        for (int w = w_lo; w <= w_hi; w++) {
            const double t0 = bench_now_sec();
            if (ds4_session_eval_batch_replay(session, prompt->v + pos, w,
                                              err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-bench: batch replay w=%d at pos %d failed: %s\n",
                        w, pos, err);
                return 1;
            }
            const double ms = (bench_now_sec() - t0) * 1000.0;
            pos += w;
            if (r >= WARMUP) {
                sum[w] += ms;
                sumsq[w] += ms * ms;
                cnt[w]++;
            }
        }
    }

    const double ms1 = cnt[1] ? sum[1] / cnt[1] : 0.0;
    for (int w = w_lo; w <= w_hi; w++) {
        const double mean = sum[w] / cnt[w];
        const double var = cnt[w] > 1
            ? (sumsq[w] - sum[w] * sum[w] / cnt[w]) / (cnt[w] - 1) : 0.0;
        fprintf(out, "%d,%d,%d,%.2f,%.2f,%.2f,%.3f\n",
                frontier, w, cnt[w], mean, var > 0.0 ? sqrt(var) : 0.0,
                1000.0 * (double)w / mean,
                ms1 > 0.0 ? ((double)w / mean) / (1.0 / ms1) : 0.0);
        fflush(out);
    }
    return 0;
}

int main(int argc, char **argv) {
    bench_config cfg = parse_options(argc, argv);

    ds4_engine_options opt = {
        .model_path = cfg.model_path,
        .backend = cfg.backend,
        .n_threads = cfg.threads,
        .prefill_chunk = cfg.prefill_chunk,
        .ssd_streaming_cache_experts = cfg.ssd_streaming_cache_experts,
        .ssd_streaming_cache_bytes = cfg.ssd_streaming_cache_bytes,
        .ssd_streaming_preload_experts = cfg.ssd_streaming_preload_experts,
        .simulate_used_memory_bytes = cfg.simulate_used_memory_bytes,
        .power_percent = cfg.power_percent,
        .warm_weights = cfg.warm_weights,
        .quality = cfg.quality,
        .ssd_streaming = cfg.ssd_streaming,
        .ssd_streaming_cold = cfg.ssd_streaming_cold,
        .expert_profile_path = cfg.expert_profile_path,
        .distributed = cfg.dist,
        .mtp_path = cfg.mtp_path,
        .mtp_draft_tokens = cfg.mtp_draft_tokens,
    };
    char dist_err[256];
    if (ds4_dist_prepare_engine_options(&cfg.dist, &opt, dist_err, sizeof(dist_err)) != 0) {
        fprintf(stderr, "ds4-bench: %s\n", dist_err);
        return 2;
    }
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) return 1;
    log_context_memory(cfg.backend, cfg.ctx_alloc, cfg.prefill_chunk);

    ds4_tokens prompt = {0};
    if (!cfg.kv_restore_path) {
        char *text = read_file(cfg.prompt_path ? cfg.prompt_path : cfg.chat_prompt_path);
        if (cfg.chat_prompt_path) {
            ds4_encode_chat_prompt(engine, cfg.system, text, DS4_THINK_NONE, &prompt);
        } else {
            ds4_tokenize_text(engine, text, &prompt);
        }
        free(text);

        if (prompt.len < cfg.ctx_max) {
            fprintf(stderr,
                    "ds4-bench: prompt has %d tokens, need at least --ctx-max=%d\n",
                    prompt.len,
                    cfg.ctx_max);
            ds4_tokens_free(&prompt);
            ds4_engine_close(engine);
            return 1;
        }
    }

    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, cfg.ctx_alloc) != 0) {
        fprintf(stderr, "ds4-bench: failed to create session\n");
        ds4_tokens_free(&prompt);
        ds4_engine_close(engine);
        return 1;
    }
    if (cfg.dist.role == DS4_DISTRIBUTED_COORDINATOR &&
        wait_distributed_route(session) != 0)
    {
        ds4_session_free(session);
        ds4_tokens_free(&prompt);
        ds4_engine_close(engine);
        return 1;
    }
    maybe_warn_distributed_step_shape(&cfg, session);

    FILE *out = stdout;
    if (cfg.csv_path) {
        out = fopen(cfg.csv_path, "wb");
        if (!out) {
            fprintf(stderr, "ds4-bench: failed to open %s: %s\n", cfg.csv_path, strerror(errno));
            ds4_session_free(session);
            ds4_tokens_free(&prompt);
            ds4_engine_close(engine);
            return 1;
        }
    }
    fprintf(out, cfg.batch_cost
            ? "ctx_tokens,width,steps,ms_mean,ms_std,agg_tps,agg_speedup_vs_w1\n"
            : "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,kvcache_bytes\n");
    fflush(out);

    const int eos = ds4_token_eos(engine);
    const bool distributed = cfg.dist.role == DS4_DISTRIBUTED_COORDINATOR;
    const bool use_mtp = cfg.mtp_path != NULL && ds4_engine_mtp_draft_tokens(engine) > 1;
    /* Optional decode-token dump for logit-equivalence cross-checks. */
    FILE *tdump = NULL;
    {
        const char *tdpath = getenv("DS4_BENCH_TOKEN_DUMP");
        if (tdpath && tdpath[0]) {
            /* Greedy token-diff is only a valid gate if the MoE down-proj is
             * deterministic: with scheduling-order atomicAdd (the n_tokens>=128
             * default) f32 rounding can flip argmax run-to-run, which has
             * false-reverted good changes (C13/C20). Force the ordered path so
             * an operator can't forget. overwrite=0 respects an explicit
             * DS4_CUDA_MOE_NO_ATOMIC_DOWN=0 for those intentionally testing the
             * nondeterministic path. */
            if (!getenv("DS4_CUDA_MOE_NO_ATOMIC_DOWN")) {
                setenv("DS4_CUDA_MOE_NO_ATOMIC_DOWN", "1", 0);
                fprintf(stderr, "ds4-bench: DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 auto-set "
                                "(token-diff requested; ordered MoE down for determinism)\n");
            }
            tdump = fopen(tdpath, "w");
            if (!tdump)
                fprintf(stderr, "ds4-bench: token dump open %s: %s\n", tdpath, strerror(errno));
        }
    }
    fprintf(stderr, "ds4-bench: decode path = %s\n",
            use_mtp ? "MTP speculative combined-forward" : "plain");
    ds4_session_snapshot snap = {0};
    char err[256];
    int previous = 0;
    int rc = 0;

    if (cfg.kv_restore_path) {
        FILE *fp = fopen(cfg.kv_restore_path, "rb");
        if (!fp) {
            fprintf(stderr, "ds4-bench: open %s: %s\n",
                    cfg.kv_restore_path, strerror(errno));
            rc = 1;
            goto cleanup;
        }
        ds4_kvstore_entry hdr = {0};
        uint32_t text_bytes = 0;
        if (!ds4_kvstore_read_header(fp, &hdr, &text_bytes)) {
            fprintf(stderr, "ds4-bench: invalid KV header in %s\n", cfg.kv_restore_path);
            fclose(fp); rc = 1; goto cleanup;
        }
        if (text_bytes && fseek(fp, (long)text_bytes, SEEK_CUR) != 0) {
            fprintf(stderr, "ds4-bench: seek past text: %s\n", strerror(errno));
            fclose(fp); rc = 1; goto cleanup;
        }
        /* Title trailer (if EXT_SESSION_TITLE) sits AFTER the payload — ignore it. */
        char load_err[160] = {0};
        if (ds4_session_load_payload(session, fp, hdr.payload_bytes,
                                     load_err, sizeof(load_err)) != 0) {
            fprintf(stderr, "ds4-bench: load_payload: %s\n",
                    load_err[0] ? load_err : "unknown");
            fclose(fp); rc = 1; goto cleanup;
        }
        fclose(fp);
        const int loaded_pos = (int)hdr.tokens;
        fprintf(stderr, "ds4-bench: restored %d tokens from %s\n",
                loaded_pos, cfg.kv_restore_path);
        cfg.ctx_start = loaded_pos;
        cfg.ctx_max = loaded_pos;
        previous = loaded_pos;
    }

    for (int frontier = cfg.ctx_start; ; frontier = next_frontier(&cfg, frontier)) {
        double prefill_sec = 0.0;
        int prefill_tokens = 0;
        ds4_tokens prefix = {
            .v = prompt.v,
            .len = frontier,
            .cap = frontier,
        };

        if (!cfg.kv_restore_path) {
            const double prefill_t0 = bench_now_sec();
            if (ds4_session_sync(session, &prefix, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-bench: prefill to %d failed: %s\n", frontier, err);
                rc = 1;
                break;
            }
            const double prefill_t1 = bench_now_sec();
            prefill_sec = prefill_t1 - prefill_t0;
            prefill_tokens = frontier - previous;
        }

        if (write_frontier_logits_json(&cfg, engine, session, frontier, previous) != 0) {
            rc = 1;
            break;
        }

        if ((cfg.gen_tokens > 0 || cfg.batch_cost) && !distributed) {
            if (ds4_session_save_snapshot(session, &snap, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-bench: snapshot at %d failed: %s\n", frontier, err);
                rc = 1;
                break;
            }
        }

        if (cfg.batch_cost && !distributed) {
            const int bc_rc = cfg.batch_check > 0
                ? run_batch_check(session, engine, &prompt, frontier, cfg.batch_check)
                : run_batch_cost(session, &prompt, frontier, out, cfg.batch_cost_w);
            if (bc_rc != 0) {
                rc = 1;
                break;
            }
            if (ds4_session_load_snapshot(session, &snap, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-bench: restore at %d failed: %s\n", frontier, err);
                rc = 1;
                break;
            }
            previous = frontier;
            if (frontier >= cfg.ctx_max) break;
            continue;
        }

        const double gen_t0 = bench_now_sec();
        const bool sampled = cfg.temperature > 0.0f;
        uint64_t rng = cfg.seed;   /* reset per frontier for reproducible sampled runs */
        int produced = 0;
        while (produced < cfg.gen_tokens) {
            if (ds4_session_pos(session) + 1 >= ds4_session_ctx(session)) {
                fprintf(stderr, "ds4-bench: generation would exceed allocated context at frontier %d\n", frontier);
                rc = 1;
                break;
            }
            /* Greedy default keeps rows comparable; --temp>0 measures the real
             * sampled decode path (spec-sampling when --mtp).  Sampled spec calls
             * pass eos=-1 so generation never early-stops (full gen_tokens/row). */
            const int token = sampled
                ? ds4_session_sample(session, cfg.temperature, 0, cfg.top_p, cfg.min_p, &rng)
                : ds4_session_argmax_excluding(session, eos);
            if (token < 0) {
                fprintf(stderr, "ds4-bench: failed to choose non-EOS token at frontier %d\n", frontier);
                rc = 1;
                break;
            }
            if (use_mtp) {
                /* Speculative decode: one batched verifier forward advances
                 * the accepted prefix (first_token + matching drafts).  Mirrors
                 * the CLI/server decode path so the bench measures the real
                 * --mtp throughput, not a separate code path. */
                int toks[17];
                const int ntok = sampled
                    ? ds4_session_eval_speculative_sample(
                        session, token, cfg.gen_tokens - produced, /*eos*/ -1,
                        cfg.temperature, 0, cfg.top_p, cfg.min_p, &rng,
                        toks, (int)(sizeof(toks) / sizeof(toks[0])), err, sizeof(err))
                    : ds4_session_eval_speculative_argmax(
                        session, token, cfg.gen_tokens - produced, eos,
                        toks, (int)(sizeof(toks) / sizeof(toks[0])), err, sizeof(err));
                if (ntok < 0) {
                    fprintf(stderr, "ds4-bench: spec decode at frontier %d failed: %s\n", frontier, err);
                    rc = 1;
                    break;
                }
                if (tdump) for (int i = 0; i < ntok; i++) fprintf(tdump, "%d\n", toks[i]);
                produced += ntok;
            } else {
                if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
                    fprintf(stderr, "ds4-bench: decode at frontier %d failed: %s\n", frontier, err);
                    rc = 1;
                    break;
                }
                if (tdump) fprintf(tdump, "%d\n", token);
                produced += 1;
            }
        }
        const double gen_t1 = bench_now_sec();
        if (rc != 0) break;

        if (cfg.gen_tokens == 0) {
            /* Pure prefill benchmark: leave the live session at the frontier. */
        } else if (distributed) {
            if (ds4_session_sync(session, &prefix, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-bench: distributed replay restore at %d failed: %s\n", frontier, err);
                rc = 1;
                break;
            }
        } else {
            if (ds4_session_load_snapshot(session, &snap, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-bench: restore at %d failed: %s\n", frontier, err);
                rc = 1;
                break;
            }
        }

        const double gen_sec = gen_t1 - gen_t0;
        fprintf(out,
                "%d,%d,%.2f,%d,%.2f,%llu\n",
                frontier,
                prefill_tokens,
                prefill_sec > 0.0 ? (double)prefill_tokens / prefill_sec : 0.0,
                produced,
                gen_sec > 0.0 ? (double)produced / gen_sec : 0.0,
                (unsigned long long)(distributed ? 0 : snap.len));
        fflush(out);

        previous = frontier;
        if (frontier >= cfg.ctx_max) break;
    }

cleanup:
    if (tdump) fclose(tdump);
    if (out != stdout) fclose(out);
    ds4_session_snapshot_free(&snap);
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
    ds4_engine_close(engine);
    return rc;
}
