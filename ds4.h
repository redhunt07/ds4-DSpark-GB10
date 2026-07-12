#ifndef DS4_H
#define DS4_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "ds4_ssd.h"

/* Public engine boundary.
 *
 * The CLI and server should treat ds4_engine as the loaded model and
 * ds4_session as one mutable inference timeline.  A session owns the live KV
 * cache and logits; callers provide full token prefixes and let
 * ds4_session_sync() reuse, extend, or rebuild the graph state.  Keep this
 * header narrow so HTTP/CLI code does not depend on tensor internals. */

typedef enum {
    DS4_BACKEND_METAL,
    DS4_BACKEND_CUDA,
    DS4_BACKEND_CPU,
} ds4_backend;

typedef enum {
    DS4_THINK_NONE,
    DS4_THINK_HIGH,
    DS4_THINK_MAX,
} ds4_think_mode;

typedef enum {
    DS4_LOG_DEFAULT,
    DS4_LOG_PREFILL,
    DS4_LOG_GENERATION,
    DS4_LOG_KVCACHE,
    DS4_LOG_TOOL,
    DS4_LOG_WARNING,
    DS4_LOG_TIMING,
    DS4_LOG_OK,
    DS4_LOG_ERROR,
} ds4_log_type;

typedef struct {
    int *v;
    int len;
    int cap;
} ds4_tokens;

typedef struct {
    int id;
    float logit;
    float logprob;
} ds4_token_score;

#define DS4_DEFAULT_TEMPERATURE 1.0f
#define DS4_DEFAULT_TOP_P 1.0f
#define DS4_DEFAULT_MIN_P 0.05f
#define DS4_DSPARK_SAME_MODEL "__DS4_DSPARK_SAME_MODEL__"

typedef enum {
    DS4_MTP_DRAFT_NONE = 0,
    DS4_MTP_DRAFT_LEGACY,
    DS4_MTP_DRAFT_DSPARK,
    DS4_MTP_DRAFT_DSPARK_NONSEQ,
} ds4_mtp_draft_kind;

typedef struct {
    uint32_t n_mtp_layers;
    uint32_t block_size;
    uint32_t noise_token_id;
    uint32_t markov_rank;
    uint32_t target_layer_ids[3];
} ds4_dspark_config;

void ds4_dspark_config_init_defaults(ds4_dspark_config *cfg);
const char *ds4_mtp_draft_kind_name(ds4_mtp_draft_kind kind);
ds4_mtp_draft_kind ds4_mtp_draft_kind_guess(bool has_e_proj, bool has_main_proj, bool has_markov_w1);
ds4_mtp_draft_kind ds4_mtp_draft_kind_guess_ex(bool has_e_proj,
                                               bool has_main_proj,
                                               bool has_markov_w1,
                                               bool markov_rank_set,
                                               uint32_t markov_rank);
bool ds4_mtp_speculative_draft_ready(ds4_mtp_draft_kind kind);
bool ds4_mtp_draft_runtime_supported(ds4_backend backend, ds4_mtp_draft_kind kind);

typedef struct ds4_engine ds4_engine;
typedef struct ds4_session ds4_session;

typedef void (*ds4_session_progress_fn)(void *ud, const char *event, int current, int total);
typedef bool (*ds4_session_cancel_fn)(void *ud);

#define DS4_SESSION_SYNC_INTERRUPTED 2

typedef enum {
    DS4_DISTRIBUTED_NONE = 0,
    DS4_DISTRIBUTED_COORDINATOR,
    DS4_DISTRIBUTED_WORKER,
} ds4_distributed_role;

typedef struct {
    uint32_t start;
    uint32_t end;
    bool has_output;
    bool set;
} ds4_distributed_layers;

typedef struct {
    ds4_distributed_role role;
    ds4_distributed_layers layers;
    const char *listen_host;
    int listen_port;
    const char *coordinator_host;
    int coordinator_port;
    uint32_t prefill_chunk;
    uint32_t prefill_window;
    uint32_t activation_bits;
    bool replay_check;
    bool debug;
} ds4_distributed_options;

typedef struct {
    const char *model_path;
    const char *mtp_path;
    const char *dspark_path;
    ds4_backend backend;
    int n_threads;
    uint32_t prefill_chunk;
    int mtp_draft_tokens;
    int dspark_draft_tokens;
    float mtp_margin;
    const char *directional_steering_file;
    const char *expert_profile_path;
    float directional_steering_attn;
    float directional_steering_ffn;
    int power_percent;
    uint32_t ssd_streaming_cache_experts;
    uint64_t ssd_streaming_cache_bytes;
    uint32_t ssd_streaming_preload_experts;
    uint64_t simulate_used_memory_bytes;
    bool warm_weights;
    bool quality;
    bool ssd_streaming;
    bool ssd_streaming_cold;
    bool inspect_only;
    bool load_slice;
    uint32_t load_layer_start;
    uint32_t load_layer_end;
    bool load_output;
    ds4_distributed_options distributed;
} ds4_engine_options;

typedef void (*ds4_token_emit_fn)(void *ud, int token);
typedef void (*ds4_generation_done_fn)(void *ud);

typedef struct {
    uint64_t total_bytes;
    uint64_t raw_bytes;
    uint64_t compressed_bytes;
    uint64_t scratch_bytes;
    uint32_t prefill_cap;
    uint32_t raw_cap;
    uint32_t comp_cap;
} ds4_context_memory;

typedef struct {
    uint8_t *ptr;
    uint64_t len;
    uint64_t cap;
} ds4_session_snapshot;

typedef struct {
    char *path;
    uint64_t bytes;
} ds4_session_payload_file;

int ds4_engine_open(ds4_engine **out, const ds4_engine_options *opt);
void ds4_engine_close(ds4_engine *e);
void ds4_engine_summary(ds4_engine *e);
int ds4_engine_vocab_size(ds4_engine *e);
int ds4_engine_power(ds4_engine *e);
int ds4_engine_set_power(ds4_engine *e, int power_percent);
const char *ds4_engine_model_name(ds4_engine *e);
int ds4_engine_layer_count(ds4_engine *e);
uint32_t ds4_engine_layer_compress_ratio(ds4_engine *e, uint32_t layer);
uint64_t ds4_engine_hidden_f32_values(ds4_engine *e);
/* Stable id for cache compatibility.  0 is the original Flash shape, so old
 * KV files with the previously-zero reserved byte remain Flash-compatible;
 * Pro and later shapes must use nonzero ids. */
int ds4_engine_model_id(ds4_engine *e);
const char *ds4_backend_name(ds4_backend backend);
bool ds4_think_mode_enabled(ds4_think_mode mode);
const char *ds4_think_mode_name(ds4_think_mode mode);
const char *ds4_think_max_prefix(void);
uint32_t ds4_think_max_min_context(void);
ds4_think_mode ds4_think_mode_for_context(ds4_think_mode mode, int ctx_size);
/* Uses the active model shape selected by ds4_engine_open(); call after opening
 * the GGUF so Flash/Pro dimensions are known. */
ds4_context_memory ds4_context_memory_estimate(ds4_backend backend, int ctx_size);
ds4_context_memory ds4_context_memory_estimate_with_prefill(
        ds4_backend backend,
        int ctx_size,
        uint32_t prefill_chunk);
bool ds4_log_is_tty(FILE *fp);
void ds4_log(FILE *fp, ds4_log_type type, const char *fmt, ...);
int ds4_engine_generate_argmax(ds4_engine *e, const ds4_tokens *prompt,
                               int n_predict, int ctx_size,
                               ds4_token_emit_fn emit,
                               ds4_generation_done_fn done,
                               void *emit_ud,
                               ds4_session_progress_fn progress,
                               void *progress_ud);
int ds4_engine_collect_imatrix(ds4_engine *e,
                               const char *dataset_path,
                               const char *output_path,
                               int ctx_size,
                               int max_prompts,
                               int max_tokens);
void ds4_engine_dump_tokens(ds4_engine *e, const ds4_tokens *tokens);
int ds4_dump_text_tokenization(const char *model_path, const char *text, FILE *fp);
int ds4_engine_head_test(ds4_engine *e, const ds4_tokens *prompt);
/* Host-only exactness self-test for the speculative-sampling math (no model/GPU
 * needed).  Returns failing-case count (0 = pass). */
int ds4_spec_sampling_selftest(void);
/* MTP combined-forward correctness gate (CUDA-only).  Runs one two-token verify
 * step through both the fast batched verify and the exact N=1 decode verify over
 * an identical (start, token0, token1), then RMS-compares the per-row logits.
 * The session must be synced to a real prefix.  Returns failed-check count
 * (0 = pass); optional out params report worst-row RMS, the pass threshold,
 * whether both rows kept top1, and the nonfinite count. */
int ds4_mtp_correctness_selftest(ds4_session *s,
                                 double *out_rms,
                                 double *out_threshold,
                                 int *out_top1_match,
                                 int *out_nonfinite);
/* MTP combined-forward self-consistency probe (CUDA-only).  Runs the same fast
 * batched n=2 verify twice on identical inputs and max-abs/RMS-diffs the logit
 * rows, isolating run-to-run nondeterminism from the algorithmic batch-vs-N=1
 * gap.  Returns failed-check count (0 = bit-stable); optional out params report
 * max-abs diff, RMS, and whether both row argmaxes stayed stable. */
int ds4_mtp_selfconsistency_selftest(ds4_session *s,
                                     double *out_maxabs,
                                     double *out_rms,
                                     int *out_top_stable);
/* CUDA per-layer tensor-equivalence gate (CUDA-only).  Teacher-forces a single-
 * token decode forward layer by layer and RMS/max-abs-diffs each GPU layer's
 * post-FFN state against the CPU reference, localizing sub-argmax drift to the
 * first diverging layer.  Returns the number of layers exceeding tolerance
 * (0 = pass); optional out params report worst RMS / max-abs, first failing
 * layer (-1 if none), and the non-finite GPU element count. */
int ds4_cuda_tensor_equivalence_selftest(ds4_session *s,
                                         double rms_tol,
                                         double max_abs_tol,
                                         double *out_worst_rms,
                                         double *out_worst_max_abs,
                                         int *out_first_fail_layer,
                                         int *out_nonfinite);
/* FastMTP harvest: per-doc batch dump target. ds4_mtp_dump_begin(base) routes
 * the prefill base-HC dump to <base> and tokens to <base>.tok (overriding the
 * legacy DS4_MTP_HC_DUMP env); ds4_mtp_dump_end() closes it. Lets a harvester
 * load the model once and loop docs with a fresh session per doc. */
void ds4_mtp_dump_begin(const char *base);
void ds4_mtp_dump_end(void);
/* Suppress chatty per-session startup logs (e.g. for the per-doc harvest loop). */
void ds4_set_quiet_logs(int q);
int ds4_engine_first_token_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_full_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_prompt_test(ds4_engine *e, const ds4_tokens *prompt, int ctx_size);

void ds4_tokens_push(ds4_tokens *tv, int token);
void ds4_tokens_free(ds4_tokens *tv);
void ds4_tokens_copy(ds4_tokens *dst, const ds4_tokens *src);
bool ds4_tokens_starts_with(const ds4_tokens *tokens, const ds4_tokens *prefix);

void ds4_tokenize_text(ds4_engine *e, const char *text, ds4_tokens *out);
void ds4_tokenize_rendered_chat(ds4_engine *e, const char *text, ds4_tokens *out);
void ds4_chat_begin(ds4_engine *e, ds4_tokens *tokens);
void ds4_encode_chat_prompt(
        ds4_engine *e,
        const char *system,
        const char *prompt,
        ds4_think_mode think_mode,
        ds4_tokens *out);
void ds4_chat_append_max_effort_prefix(ds4_engine *e, ds4_tokens *tokens);
void ds4_chat_append_message(ds4_engine *e, ds4_tokens *tokens, const char *role, const char *content);
void ds4_chat_append_assistant_prefix(ds4_engine *e, ds4_tokens *tokens, ds4_think_mode think_mode);

char *ds4_token_text(ds4_engine *e, int token, size_t *len);
int ds4_token_bos(ds4_engine *e);
int ds4_token_eos(ds4_engine *e);
int ds4_token_pad(ds4_engine *e);
int ds4_token_dsml(ds4_engine *e);
int ds4_token_think_start(ds4_engine *e);
int ds4_token_think_end(ds4_engine *e);
int ds4_token_user(ds4_engine *e);
int ds4_token_assistant(ds4_engine *e);

int ds4_session_create(ds4_session **out, ds4_engine *e, int ctx_size);
void ds4_session_free(ds4_session *s);
int ds4_session_power(ds4_session *s);
int ds4_session_set_power(ds4_session *s, int power_percent);
bool ds4_session_is_distributed(ds4_session *s);

/* Runtime directional-steering control (GPU backend).  Steering scale is read
 * fresh every forward, so changes take effect on the next eval; scale 0 on both
 * axes is bit-identical to no steering (the projection is skipped).
 *  - set_steering_scale: change attn/ffn strength (direction vectors must already
 *    be loaded — at launch via --dir-steering-file, or via reload_steering).
 *  - get_steering: read current scales + whether vectors are loaded.
 *  - reload_steering: (re)load direction vectors from `path` and set scales.  A
 *    non-empty path force-loads the vectors even at scale 0 (so a profile can be
 *    staged for later per-request activation); an empty/NULL path only sets the
 *    scales.  Returns 0 on success, nonzero on load failure (err filled). */
int ds4_session_set_steering_scale(ds4_session *s, float attn, float ffn);
void ds4_session_get_steering(ds4_session *s, float *attn, float *ffn, bool *loaded);
/* True if profile `name` is already resident in the session's steering cache. */
bool ds4_session_steering_is_cached(ds4_session *s, const char *name);
/* Select profile `name` as active (loading it from `path` into a per-graph cache
 * on first use, so repeat selections are a pointer swap) and set the scales.  An
 * empty/NULL name turns steering off.  On load failure the active profile is
 * left unchanged.  reload_steering is a path-based wrapper (name = basename). */
int ds4_session_steering_select(ds4_session *s, const char *name, const char *path,
                                float attn, float ffn, char *err, size_t errlen);
int ds4_session_reload_steering(ds4_session *s, const char *path,
                                float attn, float ffn, char *err, size_t errlen);
void ds4_session_set_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud);
/* UI-only progress. It may report fine-grained progress inside a prefill chunk;
 * callers must not treat it as a durable KV checkpoint boundary. */
void ds4_session_set_display_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud);
/* Optional cooperative cancellation.  ds4_session_sync() checks it only at
 * safe boundaries where the live checkpoint is either unchanged or represents a
 * valid token prefix, and returns DS4_SESSION_SYNC_INTERRUPTED when it stops. */
void ds4_session_set_cancel(ds4_session *s, ds4_session_cancel_fn fn, void *ud);
void ds4_session_report_progress(ds4_session *s, const char *event, int current, int total);
/* Distributed coordinator sessions return 1 when the full layer route is
 * available, 0 when it is still incomplete, and -1 for a local API error. */
int ds4_session_distributed_route_ready(ds4_session *s, char *err, size_t errlen);

typedef enum {
    DS4_SESSION_REWRITE_ERROR = -1,
    DS4_SESSION_REWRITE_OK = 0,
    /* The live backend state cannot be rewritten safely in place.  The caller should
     * restore an older checkpoint if it has one, then sync to the prompt. */
    DS4_SESSION_REWRITE_REBUILD_NEEDED = 1,
} ds4_session_rewrite_result;

/* Synchronize the live session to a full prompt token prefix.  If the current
 * checkpoint is a prefix, only the suffix is evaluated; otherwise the backend
 * state is refilled from scratch. */
int ds4_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen);
bool ds4_session_rewrite_requires_rebuild(int live_len, int canonical_len, int common);
ds4_session_rewrite_result ds4_session_rewrite_from_common(
        ds4_session *s, const ds4_tokens *prompt, int common,
        char *err, size_t errlen);
int ds4_session_common_prefix(ds4_session *s, const ds4_tokens *prompt);
int ds4_session_argmax(ds4_session *s);
int ds4_session_argmax_excluding(ds4_session *s, int excluded_id);
int ds4_session_argmax_excluding_ids(ds4_session *s, const int *excluded_ids, int excluded_count);
int ds4_sample_logits(const float *logits, int n_vocab, float temperature,
                      int top_k, float top_p, float min_p, uint64_t *rng);
int ds4_session_sample(ds4_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng);
int ds4_session_sample_excluding(ds4_session *s, float temperature, int top_k,
                                 float top_p, float min_p, uint64_t *rng,
                                 int excluded_id);
int ds4_session_sample_excluding2(ds4_session *s, float temperature, int top_k,
                                  float top_p, float min_p, uint64_t *rng,
                                  int excluded_id1, int excluded_id2);
int ds4_session_top_logprobs(ds4_session *s, ds4_token_score *out, int k);
int ds4_session_token_logprob(ds4_session *s, int token, ds4_token_score *out);
int ds4_session_copy_logits(ds4_session *s, float *out, int cap);
int ds4_session_set_logits(ds4_session *s, const float *logits, int n);
int ds4_session_eval(ds4_session *s, int token, char *err, size_t errlen);
int ds4_session_eval_speculative_argmax(ds4_session *s, int first_token,
                                        int max_tokens, int eos_token,
                                        int *accepted, int accepted_cap,
                                        char *err, size_t errlen);
/* Distribution-preserving speculative SAMPLING decode (MTP at temperature > 0): drafts
 * are sampled from the MTP draft distribution and verified by rejection sampling, so the
 * committed stream is distributed exactly as plain sampling from the (truncated) target.
 * first_token is the caller's freshly-sampled token; rng/params must match the sampler
 * the caller uses.  Returns committed token count (>=1), or -1 on error. */
int ds4_session_eval_speculative_sample(ds4_session *s, int first_token,
                                        int max_tokens, int eos_token,
                                        float temperature, int top_k, float top_p, float min_p,
                                        uint64_t *rng,
                                        int *accepted, int accepted_cap,
                                        char *err, size_t errlen);
/* Spike instrumentation: teacher-force n_tokens KNOWN tokens through one
 * combined batched verify forward (multi-stream batched-decode cost model;
 * see ds4-bench --batch-cost).  Requires an MTP-loaded session (scratch
 * buffers only — the draft head is not called).  Advances the session by
 * n_tokens positions; logits = after the last token.  Width cap 5. */
int ds4_session_eval_batch_replay(ds4_session *s, const int *tokens, int n_tokens,
                                  char *err, size_t errlen);
void ds4_session_invalidate(ds4_session *s);
void ds4_session_rewind(ds4_session *s, int pos);
int ds4_session_pos(ds4_session *s);
int ds4_session_ctx(ds4_session *s);
int ds4_session_prefill_cap(ds4_session *s);
int ds4_engine_routed_quant_bits(ds4_engine *e);
bool ds4_engine_has_output_head(ds4_engine *e);
ds4_mtp_draft_kind ds4_engine_mtp_draft_kind(ds4_engine *e);
bool ds4_engine_has_mtp(ds4_engine *e);
bool ds4_engine_has_dspark(ds4_engine *e);
int ds4_engine_mtp_draft_tokens(ds4_engine *e);
int ds4_engine_dspark_draft_tokens(ds4_engine *e);
int ds4_engine_spec_draft_tokens(ds4_engine *e);
const ds4_tokens *ds4_session_tokens(ds4_session *s);

/* Low-level graph slice entry points used by distributed inference.  The
 * transport/session routing logic lives in ds4_distributed.c. */
int ds4_session_layer_slice_reset(ds4_session *s, char *err, size_t errlen);
int ds4_session_eval_layer_slice(ds4_session *s,
                                 const int *tokens,
                                 uint32_t n_tokens,
                                 uint32_t pos0,
                                 uint32_t layer_start,
                                 uint32_t layer_end,
                                 const float *input_hc,
                                 float *output_hc,
                                 bool output_logits,
                                 float *logits,
                                 char *err,
                                 size_t errlen);
int ds4_session_eval_output_head_from_hc(ds4_session *s,
                                         const float *hidden_hc,
                                         uint32_t n_tokens,
                                         float *logits,
                                         char *err,
                                         size_t errlen);

/* Disk KV payload helpers.  HTTP/agent code owns the outer file header and
 * persistence policy; the engine owns the DS4-specific serialized graph state. */
#define DS4_SESSION_PAYLOAD_MAGIC UINT32_C(0x34565344) /* "DSV4" */
#define DS4_SESSION_PAYLOAD_VERSION UINT32_C(2)
#define DS4_SESSION_PAYLOAD_U32_FIELDS 13u
#define DS4_SESSION_LAYER_PAYLOAD_MAGIC UINT32_C(0x4c565344) /* "DSVL" */
#define DS4_SESSION_LAYER_PAYLOAD_VERSION UINT32_C(1)
#define DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS 14u

uint64_t ds4_session_payload_bytes(ds4_session *s);
int ds4_session_stage_payload(ds4_session *s, ds4_session_payload_file *out,
                              char *err, size_t errlen);
int ds4_session_write_staged_payload(const ds4_session_payload_file *payload,
                                     FILE *fp, char *err, size_t errlen);
void ds4_session_payload_file_free(ds4_session_payload_file *payload);
int ds4_session_save_payload(ds4_session *s, FILE *fp, char *err, size_t errlen);
int ds4_session_load_payload(ds4_session *s, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen);
int ds4_session_save_snapshot(ds4_session *s, ds4_session_snapshot *snap, char *err, size_t errlen);
int ds4_session_load_snapshot(ds4_session *s, const ds4_session_snapshot *snap, char *err, size_t errlen);
void ds4_session_snapshot_free(ds4_session_snapshot *snap);

uint64_t ds4_session_layer_payload_bytes(ds4_session *s,
                                         uint32_t layer_start,
                                         uint32_t layer_end);
int ds4_session_save_layer_payload(ds4_session *s, FILE *fp,
                                   uint32_t layer_start, uint32_t layer_end,
                                   char *err, size_t errlen);
int ds4_session_load_layer_payload(ds4_session *s, FILE *fp,
                                   uint64_t payload_bytes,
                                   const int *tokens, uint32_t n_tokens,
                                   uint32_t layer_start, uint32_t layer_end,
                                   char *err, size_t errlen);

#endif
