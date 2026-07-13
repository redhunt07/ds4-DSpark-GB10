#ifndef DS4_PROTOCOL_STATE_H
#define DS4_PROTOCOL_STATE_H

#include <stdbool.h>
#include <stdint.h>

/* Provider-independent lifecycle for one HTTP inference request.  Protocol
 * adapters may differ in their wire format, but they must all move through
 * this state machine and publish exactly one terminal outcome. */
typedef enum {
    DS4_PROTOCOL_QUEUED = 0,
    DS4_PROTOCOL_PREFILL,
    DS4_PROTOCOL_DECODE,
    DS4_PROTOCOL_RECOVERY,
    DS4_PROTOCOL_SERIALIZE,
    DS4_PROTOCOL_COMPLETE,
    DS4_PROTOCOL_ERROR,
    DS4_PROTOCOL_CANCELLED,
} ds4_protocol_phase;

typedef struct {
    uint64_t request_id;
    ds4_protocol_phase phase;
    bool tools_enabled;
    bool recovery_attempted;
    bool terminal;
    int prompt_tokens;
    int completion_tokens;
    double created_at;
    double last_progress_at;
} ds4_protocol_state;

void ds4_protocol_state_init(ds4_protocol_state *state, uint64_t request_id,
                             bool tools_enabled, double now);
bool ds4_protocol_transition(ds4_protocol_state *state,
                             ds4_protocol_phase next, double now);
bool ds4_protocol_begin_recovery(ds4_protocol_state *state, double now);
bool ds4_protocol_finish(ds4_protocol_state *state,
                         ds4_protocol_phase terminal, double now);
void ds4_protocol_progress(ds4_protocol_state *state, int prompt_tokens,
                           int completion_tokens, double now);
bool ds4_protocol_is_terminal(ds4_protocol_phase phase);
const char *ds4_protocol_phase_name(ds4_protocol_phase phase);

#endif
