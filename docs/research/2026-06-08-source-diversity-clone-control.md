# Source-Diversity Clone Control

Date: 2026-06-08

## Summary

The generated benchmark source-diversity gate needed a clone control. A source
label alone is not evidence of independent authorship: the same pair texts can
be copied into a second source bucket and make source-count/fault-coverage gates
look healthy.

This run adds a stricter cross-source duplicate-text gate and tests three local
conditions over one generated-fault variant:

| Condition | Pairs | Sources | Shared fault groups | Cross-source duplicate text pairs | Ready |
| --- | ---: | ---: | ---: | ---: | --- |
| single-source template | 30 | 1 | 20 | 0 | false |
| metadata-cloned two-source | 60 | 2 | 20 | 30 | false |
| text-diverse two-style source | 60 | 2 | 20 | 0 | true |

The important negative result is the middle row: metadata-only source diversity
is not enough. It has two source labels and full shared fault coverage, but every
paired text is duplicated across sources. The audit now rejects this condition
with the `cross_source_duplicate_text_pairs` gate.

## Interpretation

This is not evidence that local deterministic style variants are as strong as
API-authored data. It is a methodology result: source-diverse benchmark claims
need both metadata diversity and text non-duplication. The next API-authored
benchmark should pass this clone control before activation extraction or
source-held-out transfer is treated as meaningful.

The practical rule is:

> A generated pseudo-cohesion source is not independent merely because its
> `source` metadata changed. The paired positive/negative texts must also avoid
> exact cross-source duplication.

## Local Run

The local experiment wrote uncommitted audit artifacts to:

`/tmp/social_cohesion_source_diversity_clone_control`

The committed artifact is this note plus the audit implementation and tests; raw
generated data remains out of git.
