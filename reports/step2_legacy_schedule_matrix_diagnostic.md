# SPEC-02 legacy schedule matrix diagnostic

This diagnostic compares the existing sequential endpoint calendar with a
step-alternated calendar for only `legacy__audit_echo`. It does not replace
`run_aligned_pair`, alter the production gate, select a tolerance, or produce
an identity verdict.

The isolated scheduler in
`formic/science/identity/schedule_diagnostic.py` enforces the following memory
contract:

- the complete path, including warmups, runs under `torch.no_grad()`;
- every endpoint forward verifies that autograd remains disabled;
- warmups use `capture=False`, delete each output immediately, and return no
  result;
- measured logits are detached and copied to CPU immediately;
- returned observations contain only hashes, scalar values, lists and maps;
- each call owns two fresh configured caches, with object and storage
  independence checked during execution;
- inactive CUDA allocator blocks are released only after a complete
  configuration, never between measured repetitions.

The first alternating warmup additionally records CUDA memory before cache
creation, after each endpoint forward, after each output deletion, and after
the complete warmup. These observations are diagnostic evidence only and do
not imply a numerical root cause.

## Balanced crossover extension

`scripts/step2_balanced_crossover.py` adds a separate ABBA/BAAB crossover for
`legacy__audit_echo` at horizon 8. It does not modify the production executor.
Each decode step executes both endpoints; ABBA repeats left/right/right/left as
the first side and BAAB uses the complementary order. Eight rounds rotate the
four RR, NN, RN and NR endpoint pairs so every `(calendar, pair)` occupies each
round-relative configuration ordinal exactly once.

The balanced run uses exactly six shared no-capture warmup pair traces total,
not six per configuration. Its 64 scheduled occurrences each receive three
measured repetitions, for 192 measured pair traces. Within one round, the
actual round-relative forward ordinal is
`((configuration_ordinal * 3) + repetition) * 16 + pair_local_forward_ordinal`,
covering `0..383`. The independently recorded process-lifetime diagnostic
forward ordinal is
`96 + round * 384 + round_relative_global_forward_ordinal`, covering measured
forwards `96..3167` within one attempt. The initial 96 positions are the six
shared warmup pair traces. This process-lifetime count covers diagnostic model
forwards only, not model-load internals.

Central contrasts do not claim that two distinct forwards share a
process-lifetime position. They match the same round-relative balanced
crossover calendar slot after the Latin design has placed every endpoint
treatment in every slot. Each contrast records both source rounds and both
distinct process-lifetime diagnostic ordinals. This scope distinction is
transparent method evidence, not a causal claim. Inactive CUDA blocks are
released only after the complete shared warmup phase or a complete measured
round.

The extension keeps logits in a non-serialised CPU bank only long enough to
compute same-slot endpoint contrasts, repeat stability, complementary calendar
inversions, and hash-only associations between outputs and ordinal position.
Those associations are observations only and make no causal attribution. JSON
records contain hashes, scalar metrics and explicit round-relative ordinals,
never tensors. Resume replays completed rounds and accepts them only if their
canonical immutable checkpoints are identical; run attempts and CUDA memory
measurements append rather than replace prior evidence.

This crossover is an isolated scheduling diagnostic. Its terminal artifact
explicitly states that it is not a SPEC-02 identity verdict, does not attribute
a cause, and does not change tolerances. A full campaign command is emitted
only when all exactness, control-stability, inversion, balance and last-two
checks pass; otherwise the campaign remains blocked. Diagnostic completion is
reported separately from campaign readiness. Even when the evidence is ready,
the existing official sequential launcher is not usable as-is: it requires an
explicit adaptation to the validated balanced calendar before campaign use.
