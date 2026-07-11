# DSpark Runtime Notes

DSpark support GGUFs with `deepseek4.dspark.carrier_kind=official-dspark` are
not standalone small draft models and are not compatible with the legacy
single-block MTP draft path.

The official DeepSeek reference path does this instead:

1. Run the target model and capture auxiliary hidden states from
   `deepseek4.dspark.target_layer_ids` (for DeepSeek V4 Flash DSpark this is
   typically `[40, 41, 42]`).
2. Build `main_hidden` by reducing each captured mHC hidden state across the HC
   dimension and concatenating those layer features.
3. In the DSpark drafter, compute `main_x = main_norm(main_proj(main_hidden))`.
4. Seed a fixed block with the accepted/bonus anchor token at slot 0 and the
   DSpark noise token in the remaining slots.
5. Run the DSpark draft stages (`mtp.0`, `mtp.1`, `mtp.2`) with their own draft
   KV/cache state and the `main_x` context.
6. Produce draft token logits through the shared target LM head; for official
   DSpark, apply the sequential Markov head between block positions.
7. Use the confidence head/scheduler to choose the verification prefix length.
8. Verify the scheduled draft block with the target model and commit only the
   accepted prefix plus target bonus token.

The DSpark paper describes this as a semi-autoregressive drafter: the expensive
draft backbone remains parallel over the block, while a lightweight sequential
Markov head injects local token-to-token dependency before sampling each draft
position. The confidence head is not the legacy MTP top-logit margin; it predicts
per-position prefix survival so the runtime can avoid verifying suffixes that
are unlikely to survive. In this fork the scheduler helper lives in
`ds4_dspark_confidence_schedule_prefix()`, but it must be driven by real
DSpark confidence-head outputs, not by target-model logits.

The old `metal_graph_eval_mtp_draft_from_hc()` path is legacy-MTP-shaped:

- input token embedding + `e_proj`
- previous hidden + `h_proj`
- one MTP block
- MTP output head

That path does not consume DSpark `main_hidden`, does not run all DSpark stages,
does not use the official Markov head during draft generation, and does not
perform confidence-scheduled verification. Running official DSpark weights
through it creates extra work without the algorithmic speedup.

Until the official DSpark drafter is implemented, the runtime keeps official
DSpark support metadata loadable but disables the legacy speculative path by
default. `DS4_DSPARK_OFFICIAL_MTP_PATH=1` re-enables the previous behavior only
for diagnostics.
