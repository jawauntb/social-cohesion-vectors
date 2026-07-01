# services/modal

TRIBE v2 brain-response inference + multimodal feature extraction on Modal GPU.

## Files

- `schemas.py` — typed payloads shared with the web app.
- `dynamics.py` — trajectory dynamics (velocity, curvature, novelty, surprise) and
  editor-note landmarks. **Trajectory, not snapshot.**
- `tribe_inference.py` — TRIBE v2 brain trajectory (gated model; deterministic stub
  until `HF_TOKEN` access is wired).
- `features.py` — transcript / visual / audio-prosody embeddings + editing rhythm.
- `pipeline.py` — `video -> features + trajectory -> FeatureRecord`, fanned out with
  `.map`.
- `serve.py` — POST endpoint for pre-publication scoring + editor notes.

## Setup

```bash
uv pip install -e .
modal token new
modal secret create hf-token HF_TOKEN=hf_xxx        # after TRIBE v2 access granted
```

## Run

```bash
modal run modal_app/tribe_inference.py::smoke        # trajectory smoke
modal run modal_app/pipeline.py::smoke               # batch feature extraction
modal deploy modal_app/serve.py                      # pre-post scoring endpoint
```

Copy the deployed endpoint URL into `MODAL_PREDICT_ENDPOINT` in the web app's env.

## Wiring the real TRIBE forward pass

The stubs are deterministic and seeded by video id so the pipeline + ablation run
end-to-end, but **do not report results off the stub**. Implement `_load_tribe` and the
forward pass in `tribe_inference.py`, and replace the stub extractors in `features.py`
with Whisper / CLIP / librosa once model access is set up.
