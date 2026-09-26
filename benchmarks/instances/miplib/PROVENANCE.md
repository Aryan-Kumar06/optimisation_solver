# MIPLIB 2017 instances

Source: https://miplib.zib.de/ — metadata from the Collection Set table
(`tag_collection.html`), instances from `WebData/instances/<name>.mps.gz`.

## Encoding

MIPLIB ships **gzip**, unlike Netlib's historical `emps` format. The encoding is
still detected from the file's own bytes rather than from the `.gz` extension,
so a source that changes format is noticed rather than silently mis-decoded.

## Reference objectives

Taken from the Collection Set table. Whether a value is a **proven optimum** or
only a **best-known incumbent** is recorded per instance in `manifest_*.json` as
`reference_kind`:

| MIPLIB status | reference_kind | meaning |
|---|---|---|
| `easy`, `hard` | `proven_optimal` | solved to proven optimality by reference solvers |
| `open` | `best_known_only` | best incumbent found; no optimality proof exists |

MIPLIB's `easy`/`hard` status describes **reference** solvers, not ours. It was
not used to select for tractability, and carries no expectation that these are
easy for this engine. Both frozen selections record this caveat explicitly.

## Selection protocol

`FROZEN_SMOKE.json` (5 instances) and `FROZEN_DEV25.json` (25 instances) were
each written **before** the first corresponding run. Criteria come from the
solver's documented limits — the dual simplex's dense *m*×*m* basis inverse
(constraints < 2000), the dispatcher's nonzero ceiling, and the reader's refusal
to represent `INDICATORS`/`SOS` sections.

Compatibility screening (does the file parse? do dimensions match the metadata?)
happens *after* freezing. It is not a performance observation, and an instance
that turns out incompatible is recorded as such rather than replaced.

## MIPLIB's official checker

**Not obtained.** `https://miplib.zib.de/downloads/checker.zip` returns an HTML
page rather than an archive, and the checker is not present in the public
`ambros-gleixner/MIPLIB2017` repository paths tried. Verification therefore
rests on `benchmarks/lib/verify.py`, which performs the equivalent checks
against the original model:

- row activities and variable bounds evaluated at the **raw returned values**,
  with no rounding applied first;
- integrality measured as `|x_j - round(x_j)|`, reported separately;
- objective recomputed from an independently written reader.

If the official checker becomes available it should be run alongside, not
instead of, this one.
