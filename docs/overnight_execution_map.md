# Overnight Execution Map

## Compute-Only Tonight

These work without new humans or new neural recordings.

1. Scenario benchmark
   - Create 20-30 social dilemmas across repeated games, public goods,
     negotiation, restorative repair, and resource allocation.
   - Produce cooperative, self-protective, and adversarial transcript variants.

2. Cohesion rubric
   - Score cooperation, repair, fairness, hostility, truthfulness, and autonomy
     safety.
   - Keep sub-scores visible so "cohesion" does not collapse into compliance.

3. Pairwise training data
   - Convert each scenario into high/low cohesion trajectory pairs.
   - Export prompts for classifiers, reward models, activation captures, and
     Prolific pilots.

4. Open-weight activation capture
   - Use Modal to run small open instruction models first.
   - Save mean-pooled residual activations per transcript/prompt.
   - Train contrastive top-vs-bottom directions as the first cohesion-vector
     baseline.

5. SAE-ready artifact export
   - Once the target model/layer is selected, export activation tensors and
     metadata in stable sample IDs.
   - Use existing SAE tooling as an analysis layer rather than blocking the
     basic vector pipeline on SAE training.

6. Best-of-N content selection
   - Generate multiple candidate interventions for a conflict scenario.
   - Score them with the rubric and activation-derived direction.
   - Filter with truthfulness/autonomy guardrails before any human study.

## Completed First Sprint

- Scripted scenario scaffold: 25 scenarios, 450 simulated runs, 126 pairwise
  examples, 252 activation prompts.
- Generated offline benchmark: 125 generated trajectories, 50 pairwise examples,
  100 activation prompts.
- Modal/open-model activation lane: Qwen 0.5B, 1.5B, 3B, and GPT-2 smoke runs.
- Transfer checks: scripted/generated pair-set transfer is wired.
- Pseudo-cohesion hard negatives: expanded to 30 pseudo cases, 30 matched
  genuine contrasts, and exportable activation prompts.
- GPT-2 failure analysis: generated misses concentrate on
  `pseudo_cohesion_compliance` negatives.
- Expanded Qwen 0.5B Modal pass: 0.967 leave-one-pair-out accuracy on the 30
  pseudo-cohesion pairs, with one failure on `resource_request`.
- GPT-2 SAE smoke: residual activations reach 0.967 leave-one-pair-out accuracy
  on the expanded pseudo set, while SAE features reach 0.533, giving a sharper
  target for feature-level inspection.
- GPT-2 SAE token inspection: 3056 remains a genuine-skew candidate;
  24555/11737/703 remain pseudo-skew candidates; 28005/20249 are demoted by
  token-level evidence.
- Expanded pseudo-cohesion SAE transfer: seed-plus-variant export creates 120
  matched pairs / 240 prompts. The inspected GPT-2 SAE feature ensemble reaches
  0.825 leave-one-pair-out accuracy with mean activations; 703 is the best
  single mean-activation feature at 0.792, while 3056 remains genuine-skewed but
  weak alone at 0.600.

## Low-Cost Human Next

These need Prolific or comparable validation but not new neural recordings.

1. Pairwise judgment pilot
   - Participants compare high-scoring vs low-scoring interventions.
   - Outcomes: trust, willingness to continue dialogue, perceived manipulation,
     fairness, willingness to cooperate.

2. Behavioral game pilot
   - Expose participants to selected interventions, then ask for cooperate/defect
     or allocation decisions.
   - Test whether model-selected content changes choices, not just ratings.

3. Active-learning loop
   - Send uncertain or model-disagreement pairs to humans.
   - Use those labels to calibrate the reward model and guardrail thresholds.

## Later / Requires Real Neural Work

1. Brain-aligned proxy bridge
   - Compare generated content in TRIBE/V-JEPA-like predicted embedding spaces.
   - Treat as hypothesis generation, not human neural evidence.

2. Existing public neural datasets
   - Reuse public fMRI/EEG/naturalistic datasets where stimuli and social labels
     fit the question.
   - Useful for alignment checks, not novel in-group cohesion.

3. New recordings
   - Individual fMRI after content exposure.
   - EEG/fNIRS hyperscanning during group negotiation.
   - n-1 or novel in-group designs.
   - Requires IRB, consent, participant recruitment, and careful claims.

## Decision Criteria

- If offline rubric and activation vectors do not separate obvious cooperative
  vs adversarial transcripts, fix the benchmark before using human time.
- If vectors separate scripted transcripts but fail LLM-generated held-out
  scenarios, improve scenario diversity and reduce lexical shortcuts.
- If weak or SAE-compatible models fail specifically on pseudo-cohesion, expand
  that dataset and inspect sparse features before naming a "cohesion" direction.
  Treat wrapper-sensitive or punctuation-heavy features as artifacts until they
  survive cleaner generated variants.
- If vectors transfer across scenarios/models, run Prolific.
- If Prolific shows human preference or behavioral movement, then the
  brain-aligned bridge becomes worth spending compute on.
