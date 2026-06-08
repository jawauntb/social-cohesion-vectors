# API Provider Auth Fail-Fast

Date: 2026-06-08

## Summary

The first live API-authored hard-negative attempt did not produce benchmark
text because both provider keys available in `/Users/jawaun/superoptimizers/.env`
were rejected by their providers:

- OpenAI: `401 invalid_api_key`
- Anthropic: `401 invalid x-api-key`

No secret values are recorded here. The raw output logs sanitize key-like
strings before writing error details.

## Process Finding

Before this change, a fatal authentication error was repeated for every prompt
in the 60-prompt one-variant contract. That made a configuration problem look
like a full generation run and wasted provider attempts.

The API generation runner now treats authentication failures as fatal for the
current run:

- one raw row records the real provider `request_error`;
- remaining prompt rows are written as
  `request_skipped_after_fatal_error`;
- the prompt contract remains complete for audit and replay;
- nonfatal request errors, such as a transient server error, still allow later
  prompts to continue.

## Local Check

The OpenAI fail-fast check wrote uncommitted artifacts to:

`/tmp/social_cohesion_api_openai_failfast_check`

Raw output status counts:

| Status | Count |
| --- | ---: |
| `request_error` | 1 |
| `request_skipped_after_fatal_error` | 59 |

This is not a benchmark result or activation finding. It is a runner-readiness
fix discovered while attempting the next API-authored hard-negative experiment.
