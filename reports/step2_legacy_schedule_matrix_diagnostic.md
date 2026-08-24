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
