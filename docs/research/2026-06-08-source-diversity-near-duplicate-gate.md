# Source-Diversity Near-Duplicate Gate

Date: 2026-06-08

## Summary

The source-diversity audit now checks for lightly edited cross-source clones, not
only exact duplicate paired texts. The new gate uses a conservative base-aware
character 5-gram Jaccard similarity threshold of 0.82 over the combined
positive/negative pair text.

The gate is base-aware: when `base_contrast_id` is available, it compares
cross-source pairs only within the same base contrast. This avoids punishing
legitimate benchmark pairs that share scenario boilerplate across different
fault classes.

## Local Probe

One generated-fault variant was evaluated under four source conditions:

| Condition | Sources | Exact duplicates | Near duplicates | Max similarity | Ready |
| --- | ---: | ---: | ---: | ---: | --- |
| metadata exact clone | 2 | 30 | 30 | 1.000000 | false |
| lightly edited clone | 2 | 0 | 30 | 0.950000 | false |
| template + cue-balanced | 2 | 0 | 0 | 0.415430 | true |
| cue-balanced + lexical-hardened | 2 | 0 | 0 | 0.259791 | true |

## Interpretation

This closes the next source-diversity loophole. A second source can no longer
pass merely by making small wording edits to the same paired texts. At the same
time, the threshold does not reject the legitimate deterministic two-style
scaffolds we have been using for local audit development.

This remains a text-level provenance gate, not evidence of model activation
behavior. Its value is methodological: future API-authored or externally
generated hard negatives must pass both exact clone control and near-duplicate
control before source-held-out activation claims are treated as meaningful.
