#include "ds4_dspark_runtime.h"

#include <math.h>
#include <string.h>

void ds4_dspark_config_init_defaults(ds4_dspark_config *cfg) {
    if (!cfg) return;
    memset(cfg, 0, sizeof(*cfg));
    cfg->n_mtp_layers = 3;
    cfg->block_size = 5;
    cfg->noise_token_id = 128799u;
    cfg->markov_rank = 256;
    cfg->target_layer_ids[0] = 40;
    cfg->target_layer_ids[1] = 41;
    cfg->target_layer_ids[2] = 42;
}

const char *ds4_mtp_draft_kind_name(ds4_mtp_draft_kind kind) {
    switch (kind) {
    case DS4_MTP_DRAFT_LEGACY: return "legacy-mtp";
    case DS4_MTP_DRAFT_DSPARK: return "dspark";
    case DS4_MTP_DRAFT_DSPARK_NONSEQ: return "dspark-nonseq";
    default: return "none";
    }
}

ds4_mtp_draft_kind ds4_mtp_draft_kind_guess_ex(bool has_e_proj,
                                               bool has_main_proj,
                                               bool has_markov_w1,
                                               bool markov_rank_set,
                                               uint32_t markov_rank) {
    if (has_main_proj && has_markov_w1) return DS4_MTP_DRAFT_DSPARK;
    if (has_main_proj && markov_rank_set && markov_rank == 0) return DS4_MTP_DRAFT_DSPARK_NONSEQ;
    if (has_e_proj) return DS4_MTP_DRAFT_LEGACY;
    return DS4_MTP_DRAFT_NONE;
}

ds4_mtp_draft_kind ds4_mtp_draft_kind_guess(bool has_e_proj,
                                            bool has_main_proj,
                                            bool has_markov_w1) {
    return ds4_mtp_draft_kind_guess_ex(has_e_proj, has_main_proj, has_markov_w1, false, 0);
}

float ds4_dspark_bf16_to_f32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

int ds4_dspark_draft_len_until_eos(const int *drafts, int draft_n, int eos_token) {
    if (!drafts || draft_n <= 0) return 0;
    for (int i = 0; i < draft_n; i++) {
        if (drafts[i] == eos_token) return i + 1;
    }
    return draft_n;
}

int ds4_dspark_prefix_slot_for_accept(int accepted, int draft_n) {
    if (accepted <= 0 || draft_n <= 1 || accepted >= draft_n) return -1;
    return accepted - 1;
}

int ds4_dspark_prefix_slot_count(ds4_mtp_draft_kind kind, int block_size, int max_slots) {
    if (max_slots <= 0) return 0;
    if (kind != DS4_MTP_DRAFT_LEGACY &&
        kind != DS4_MTP_DRAFT_DSPARK &&
        kind != DS4_MTP_DRAFT_DSPARK_NONSEQ) {
        return 0;
    }
    int slots = 1;
    if (kind == DS4_MTP_DRAFT_DSPARK || kind == DS4_MTP_DRAFT_DSPARK_NONSEQ) {
        slots = block_size > 1 ? block_size - 1 : 1;
    }
    if (slots > max_slots) slots = max_slots;
    return slots;
}

int ds4_dspark_confidence_schedule_prefix(const float *confidence,
                                          int block_size,
                                          int max_prefix,
                                          float min_survival,
                                          float verify_cost_per_token) {
    if (!confidence || block_size <= 0 || max_prefix <= 0) return 0;
    if (max_prefix > block_size) max_prefix = block_size;
    if (!(min_survival > 0.0f && min_survival <= 1.0f)) min_survival = 0.50f;
    (void)verify_cost_per_token;

    for (int i = 0; i < max_prefix; i++) {
        float c = confidence[i];
        float p;
        if (c >= 16.0f) {
            p = 1.0f;
        } else if (c <= -16.0f) {
            p = 0.0f;
        } else {
            p = 1.0f / (1.0f + expf(-c));
        }
        if (p < min_survival) return i;
    }
    return max_prefix;
}

ds4_dspark_spec_gate ds4_dspark_speculative_gate(ds4_mtp_draft_kind kind,
                                                 bool mtp_ready,
                                                 int mtp_draft_tokens) {
    if (!mtp_ready || mtp_draft_tokens <= 1) return DS4_DSPARK_SPEC_DISABLED;
    if (kind == DS4_MTP_DRAFT_LEGACY) return DS4_DSPARK_SPEC_LEGACY_MTP;
    if (kind == DS4_MTP_DRAFT_DSPARK) return DS4_DSPARK_SPEC_DSPARK_ENABLED;
    if (kind == DS4_MTP_DRAFT_DSPARK_NONSEQ) return DS4_DSPARK_SPEC_DSPARK_NONSEQ_NOT_READY;
    return DS4_DSPARK_SPEC_DISABLED;
}

const char *ds4_dspark_spec_gate_reason(ds4_dspark_spec_gate gate) {
    switch (gate) {
    case DS4_DSPARK_SPEC_LEGACY_MTP:
        return "legacy MTP draft path (DSpark block draft not engaged)";
    case DS4_DSPARK_SPEC_DSPARK_ENABLED:
        return "DSpark block speculative decode enabled";
    case DS4_DSPARK_SPEC_DSPARK_NOT_READY:
        return "DSpark draft graph has not been validated on real DSpark GGUF weights; speculative decode stays off (no fake draft tokens)";
    case DS4_DSPARK_SPEC_DSPARK_NONSEQ_NOT_READY:
        return "DSpark nonseq draft head has not been validated on real trained DSpark GGUF weights; speculative decode stays off (no fake draft tokens)";
    case DS4_DSPARK_SPEC_DISABLED:
    default:
        return "speculative draft disabled";
    }
}

bool ds4_mtp_speculative_draft_ready(ds4_mtp_draft_kind kind) {
    return kind == DS4_MTP_DRAFT_LEGACY ||
           kind == DS4_MTP_DRAFT_DSPARK ||
           kind == DS4_MTP_DRAFT_DSPARK_NONSEQ;
}

bool ds4_mtp_draft_runtime_supported(ds4_backend backend, ds4_mtp_draft_kind kind) {
    if (backend == DS4_BACKEND_CPU) return false;
    if (!ds4_mtp_speculative_draft_ready(kind)) return false;
    return true;
}
