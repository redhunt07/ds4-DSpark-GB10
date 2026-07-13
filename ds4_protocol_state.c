#include "ds4_protocol_state.h"

#include <string.h>

bool ds4_protocol_is_terminal(ds4_protocol_phase phase) {
    return phase == DS4_PROTOCOL_COMPLETE || phase == DS4_PROTOCOL_ERROR ||
           phase == DS4_PROTOCOL_CANCELLED;
}

static bool transition_allowed(ds4_protocol_phase from,
                               ds4_protocol_phase to) {
    if (ds4_protocol_is_terminal(from)) return false;
    if (ds4_protocol_is_terminal(to)) return true;
    switch (from) {
    case DS4_PROTOCOL_QUEUED:
        return to == DS4_PROTOCOL_PREFILL || to == DS4_PROTOCOL_SERIALIZE;
    case DS4_PROTOCOL_PREFILL:
        return to == DS4_PROTOCOL_DECODE || to == DS4_PROTOCOL_SERIALIZE;
    case DS4_PROTOCOL_DECODE:
        return to == DS4_PROTOCOL_RECOVERY || to == DS4_PROTOCOL_SERIALIZE;
    case DS4_PROTOCOL_RECOVERY:
        return to == DS4_PROTOCOL_DECODE || to == DS4_PROTOCOL_SERIALIZE;
    case DS4_PROTOCOL_SERIALIZE:
        return false;
    case DS4_PROTOCOL_COMPLETE:
    case DS4_PROTOCOL_ERROR:
    case DS4_PROTOCOL_CANCELLED:
        return false;
    }
    return false;
}

void ds4_protocol_state_init(ds4_protocol_state *state, uint64_t request_id,
                             bool tools_enabled, double now) {
    memset(state, 0, sizeof(*state));
    state->request_id = request_id;
    state->phase = DS4_PROTOCOL_QUEUED;
    state->tools_enabled = tools_enabled;
    state->created_at = now;
    state->last_progress_at = now;
}

bool ds4_protocol_transition(ds4_protocol_state *state,
                             ds4_protocol_phase next, double now) {
    if (!state || !transition_allowed(state->phase, next)) return false;
    state->phase = next;
    state->last_progress_at = now;
    return true;
}

bool ds4_protocol_begin_recovery(ds4_protocol_state *state, double now) {
    if (!state || state->recovery_attempted ||
        state->phase != DS4_PROTOCOL_DECODE)
        return false;
    state->recovery_attempted = true;
    return ds4_protocol_transition(state, DS4_PROTOCOL_RECOVERY, now);
}

bool ds4_protocol_finish(ds4_protocol_state *state,
                         ds4_protocol_phase terminal, double now) {
    if (!state || !ds4_protocol_is_terminal(terminal) || state->terminal)
        return false;
    if (!ds4_protocol_transition(state, terminal, now)) return false;
    state->terminal = true;
    return true;
}

void ds4_protocol_progress(ds4_protocol_state *state, int prompt_tokens,
                           int completion_tokens, double now) {
    if (!state || state->terminal) return;
    if (prompt_tokens >= 0) state->prompt_tokens = prompt_tokens;
    if (completion_tokens >= 0) state->completion_tokens = completion_tokens;
    state->last_progress_at = now;
}

const char *ds4_protocol_phase_name(ds4_protocol_phase phase) {
    switch (phase) {
    case DS4_PROTOCOL_QUEUED: return "queued";
    case DS4_PROTOCOL_PREFILL: return "prefill";
    case DS4_PROTOCOL_DECODE: return "decode";
    case DS4_PROTOCOL_RECOVERY: return "recovery";
    case DS4_PROTOCOL_SERIALIZE: return "serialize";
    case DS4_PROTOCOL_COMPLETE: return "complete";
    case DS4_PROTOCOL_ERROR: return "error";
    case DS4_PROTOCOL_CANCELLED: return "cancelled";
    }
    return "unknown";
}
