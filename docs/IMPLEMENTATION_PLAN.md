# THALAMUS — Implementation Plan

This document identifies what is currently built, what the doc describes as gaps to real ML, and provides a step-by-step plan to implement everything the slides describe.

---

## Part 1 — What Is Currently Developed

The following is what exists in code today, as described in the main body of THALAMUS_SLIDES.md (excluding the "Gap to real ML" sections).

### Path A — Evolutionary Selector

**Stage 1 — Component Scoring** (`component_scoring/`)
- Scanners for skills (SKILL.md), memory (## headings), tools (AST Tool* classes)
- ComponentRecord data structure
- SHA-256 fingerprint change detection — skip unchanged components
- LLM call 1: query generator (20 pairs per component, temperature 0.8)
- LLM call 2: evaluator (one call per pair, component as system prompt)
- Four lexical metrics: F1, Bigram F1, Bag-of-Words, Length Ratio
- Parallel evaluation (up to 5 workers)
- Output: `scoring_matrix_TYPE_NAME.json` per component

**Stage 2 — Score Enrichment** (`component_scoring/enrichment/`)
- Reads `turns_YYYY-WNN.jsonl`
- Bayesian blending: `synthetic_weight = max(0, 1 − n_real / n_needed)`
- Updates `mean_score` field in scoring matrices
- Optional — system works without it

**Stage 3 — Query Clustering** (`oracle_builder/evolutionary/config_builder_step03_fit_clusters.py`)
- TF-IDF backend: 2000-feature vocabulary, K-means K=20
- Sentence-transformer backend: `all-MiniLM-L6-v2`, K-means K=20
- Per-component query centroid (mean embedding over example inputs)
- Saves fitted clusterer to `context_configs.pkl`

**Stage 4 — Evolutionary Search** (`oracle_builder/evolutionary/evolution/`)
- 60 optimization targets (20 clusters × 3 budget tiers)
- Genome: N-bit bitmask (N = total components)
- Fitness: `Σᵢ[mean_score_i × cosine(v, centroid_i) × bit_i] − 0.1 × tokens/budget`
- Over-budget penalty
- Tournament selection (k=3), uniform crossover, bit-flip mutation (p=0.05)
- 200 generations, population 100
- Pareto front extraction (non-dominated by fitness and tokens)
- Optional LLM Pareto validation (`--validate-pareto`)
- Component relevance ordering (most-relevant-first)

**Stage 5 — Write Output** (`oracle_builder/evolutionary/config_builder_step07_serialize_output.py`)
- `context_configs.json`: optimal configs per cluster × budget
- `context_configs.pkl`: serialized clusterer

**Path A Query Time — ClusterSelector** (`context_selectors/cluster_selector.py`)
- Vectorize query, K-means predict nearest cluster
- JSON lookup of optimal config
- Three ordering strategies: `relevance`, `bookend`, `none`
- `select_auto()` with BudgetEstimator
- Under 10 ms latency

### Path B — Classifier Path

**Turn Logging** (`shared/turn_logger.py`)
- Logs per turn: `query_embedding`, `context_config`, `outcome`, `exploration`
- Weekly JSONL files: `turns_YYYY-WNN.jsonl`
- Privacy: stores embedding vector, not raw query text
- Off-policy exploration: adds random components at `exploration_rate`

**Outcome Scoring** (`shared/outcome_scorer.py`)
- `quality = 0.5 + 0.20·completed − 0.30·correction + max(0, 0.10 − 0.02·length)`
- Clamped to [0.0, 1.0]
- Explicit override: "positive" → 1.0, "negative" → 0.0

**Classifier Training** (`oracle_builder/classifier/component_classifier_trainer.py`)
- N independent logistic regression models (one per component)
- Features: `query_embedding`; label: 1 if quality > 0.5
- L2 regularization, C=1.0, solver lbfgs
- Minimum data guard: `min_turns` (default 10)
- Recency window: last 8 weeks
- Off-policy explored turns included in training
- Output: `classifier.pkl` (W matrix N×d, bias N-vector, component names)

**Path B Query Time — ClassifierSelector** (`context_selectors/classifier_selector.py`)
- `p_i = σ(W_i · e + b_i)` per component
- Include if p_i > 0.5
- Confidence: `mean(max(p, 1−p))`
- Under 1 ms latency

### Shared Infrastructure

- `BudgetEstimator`: word count + multi-step regex heuristic → small/medium/large
- `bookend_order()`: rearranges relevance-sorted list to place most-relevant at context edges
- `QueryClusterer`: dual TF-IDF / sentence-transformer backend
- `ComponentInclusionClassifier`: linear model with pickle serialization
- CLI entry points for all three packages

---

## Part 2 — Gaps to Real ML

These are extracted from every "Gap to real ML" collapsible section in THALAMUS_SLIDES.md. They are organized by slide.

### From Slide 9 — Stage 1 Scoring
1. **Synthetic expected answers** — the query generator covers what the LLM thinks is representative, not the real production distribution
2. **Lexical metrics are not semantic correctness** — F1, bigram F1, bag-of-words penalize correct paraphrases and reward plausible-sounding wrong text. BERTScore or an LLM judge would be substantially more accurate
3. **Isolated evaluation** — Stage 1 puts only one component in context. A component that is mediocre alone but excellent in combination receives a low synthetic score and is deprioritized by the GA

### From Slide 15 — Fitness Function
4. **λ = 0.1 is a fixed guess** — the right token penalty weight differs per cluster, per deployment, and over time
5. **The formula structure is hand-crafted, not learned** — the multiplication of mean_score × cosine, linear sum over components, and linear token penalty are design choices not validated against real data
6. **Linear fitness cannot detect combination effects** — two components jointly necessary but individually modest each get modest rewards; the GA cannot discover they should always appear together

### From Slide 25 — Classifier Architecture
7. **N independent classifiers cannot model joint necessity** — if skill A and tool B are useful only together, both classifiers learn low correlation with good outcomes individually and the pair gets excluded
8. **Fix requires joint modeling** — a multi-label classifier (one model, N outputs), or a model that takes currently-selected components as additional features, or a sequential selection model

### From Slide 28 — Outcome Labels
9. **task_completed is coarse** — a perfect answer and a barely-adequate one both get quality ≈ 0.70
10. **follow_up_correction conflates causes** — the user corrected because of bad context selection OR bad model reasoning; the label blames context selection regardless
11. **conversation_length is a proxy of a proxy** — a complex task legitimately requires a long conversation; a short conversation may mean the user gave up
12. **Explicit ratings are rare** — almost all training examples use the noisy implicit formula

### From Slide 30 — Training Requirements
13. **No train/test split or cross-validation** — system cannot answer "is the new classifier better than the old one?" before deploying it
14. **No performance tracking over time** — no dashboard showing whether classifier accuracy is improving, which components have poor precision, or whether Path B outperforms Path A
15. **No model versioning or rollback** — `classifier.pkl` is overwritten in place; no way to revert if retraining produces a worse model
16. **No automated hyperparameter tuning** — C=1.0 and threshold=0.5 are fixed for all components; rare components need stronger regularization, common ones less

### From Slide 32 — Feedback Loop
17. **Classifier does not use cluster oracle as a prior** — when Path B starts with 10–50 turns per component, it learns from scratch rather than warm-starting from Path A's relevance scores
18. **GA fitness function does not use classifier signal** — after hundreds of turns the classifier contains real-data signal about query–component correlations; the GA's hand-crafted formula ignores it

### From Slide 33 — System Maturity
19. **No concept drift detection** — if query distribution shifts, neither path detects it or triggers retraining automatically
20. **No oracle staleness detection** — if the skill library changes significantly, `context_configs.json` may recommend components that no longer exist or miss new ones
21. **No automatic retraining schedule** — operator must manually run `train-classifier` and `evolve`
22. **Fixed transition thresholds** — 100 and 500 turns are guesses; the right threshold depends on number of components, query variety, and label noise

### From Slide 35 — Parameter Tuning
23. **C = 1.0 not tuned** — dataset size and noise level determine the right regularization strength
24. **threshold = 0.5 not tuned per component** — precision/recall tradeoff differs by component and deployment
25. **K = 20 not tuned** — spurious clusters if too high; lumped subtypes if too low
26. **n_needed = 100 not tuned** — noisy signals need more data; clean signals can take over earlier

---

## Part 3 — Step-by-Step Implementation Plan

The plan follows the ordering from Slide 39 ("Road to AutoML"). Each step must be completed before the next begins.

---

### Step 1 — Better Outcome Labels

**Addresses gaps:** 9, 10, 11, 12

**Why first:** every downstream component (Stage 2 enrichment, classifier training, hyperparameter tuning, evaluation) depends on the quality of the outcome label. Fixing labels before building anything else avoids training and validating on a noisy signal.

**What to build:**

1a. Add an **LLM judge scorer** to `shared/outcome_scorer.py`:
- New method `score_from_llm_judge(query, agent_output, reference_output, model, api_key) → float`
- Prompt: given the query and reference, rate the agent's output on a 1–10 scale
- Normalize to [0, 1]: `(score − 1) / 9`
- Returns `None` if judge call fails so the implicit formula is used as fallback

1b. Extend `TurnLogger.log_turn()` to accept an optional `llm_judge_score: float` parameter:
- If provided, store it in the `outcome` field as `"llm_judge_score"`
- `OutcomeScorer.compute()` uses it preferentially over the implicit formula if present

1c. Extend `OutcomeScorer.compute()` priority chain:
- Priority 1: `explicit_rating` (positive/negative)
- Priority 2: `llm_judge_score` (if provided)
- Priority 3: implicit formula (existing behavior, unchanged)

1d. Add `--judge-model` and `--judge-api-key` flags to the oracle_builder CLI so operators can enable LLM judging when scoring enrichment runs.

**Verification:** run Stage 2 enrichment with both the old and new scorer on the same logs; confirm that `updated_mean_score` values differ and that the LLM judge path can be toggled.

---

### Step 2 — Held-out Evaluation Set and Model Versioning

**Addresses gaps:** 13, 14, 15

**Why second:** before tuning any parameter, the system needs a measurement framework. Tuning without evaluation is guessing.

**What to build:**

2a. Add a **log splitter** in `oracle_builder/classifier/`:
- `log_splitter.py`: reads all JSONL files in the recency window
- Splits by turn timestamp: most recent 20% of turns → held-out validation set; remaining 80% → training set
- Split is deterministic (sorted by timestamp, not random) so the held-out set always contains the most recent behavior

2b. Add a **classifier evaluator** in `oracle_builder/classifier/`:
- `classifier_evaluator.py`: loads `classifier.pkl` and a held-out split
- Computes per-component precision, recall, F1, and AUC
- Computes aggregate: macro-averaged F1, mean AUC across all components
- Writes `classifier_eval_YYYY-MM-DD.json` to the oracle directory

2c. Add a **model registry** replacing the single `classifier.pkl`:
- Each training run writes `classifier_YYYY-MM-DD_HHMMSS.pkl`
- A `classifier_registry.json` file tracks: filename, training date, training turn count, validation F1, validation AUC
- A symlink or pointer `classifier_current.pkl` points to the currently active model
- `ClassifierSelector` loads from `classifier_current.pkl`

2d. Add a **promotion gate** to `cmd_train_classifier.py`:
- After training, evaluate the new model on the held-out set
- Only update `classifier_current.pkl` if the new model's validation F1 exceeds the current model's stored F1 by at least 0.01
- If the gate fails, log the result and keep the previous model active
- Add a `--force-promote` flag to bypass the gate when needed

2e. Add a `list-versions` subcommand to the oracle_builder CLI:
- Prints the classifier registry in a human-readable table (date, turns, F1, AUC, active)

**Verification:** train two classifiers on the same logs, confirm the registry records both, confirm the worse one is not promoted.

---

### Step 3 — Hyperparameter Tuning

**Addresses gaps:** 16, 23, 24, 25, 26

**Why third:** now that we can measure whether a model is better or worse (Step 2), we can tune parameters with a real signal.

**What to build:**

3a. Add a **hyperparameter search module** in `oracle_builder/classifier/`:
- `hyperparameter_search.py`
- Parameter grid:
  - `C`: [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
  - `threshold`: [0.3, 0.4, 0.5, 0.6, 0.7]
- For each candidate (C, threshold): train on the 80% split, evaluate on the held-out 20% split
- Select the (C, threshold) pair that maximizes macro F1 on the held-out set
- Per-component thresholds: after finding the best global threshold, run a per-component sweep over [0.3, 0.5, 0.7] and keep per-component values

3b. Persist tuned hyperparameters in `classifier_registry.json`:
- Store `best_C`, `per_component_thresholds` alongside the model file path

3c. Update `ClassifierSelector` to load per-component thresholds from the registry instead of using the fixed 0.5 default.

3d. Add a **cluster count tuner** in `oracle_builder/evolutionary/`:
- `cluster_count_tuner.py`
- Fits K-means for K in [5, 10, 15, 20, 30, 50]
- Computes inertia (within-cluster sum of squares) and silhouette score for each K
- Selects K at the elbow of the inertia curve (first derivative minimum)
- Adds `--auto-k` flag to the `evolve` command; if set, runs the tuner before clustering

3e. Add a **λ tuner per cluster** in `oracle_builder/evolutionary/`:
- `lambda_tuner.py`
- After at least 100 logged turns exist, for each cluster compute the correlation between `config.fitness` (using different λ values) and mean outcome quality of turns assigned to that cluster
- Select the λ value that maximizes this correlation
- Store per-cluster λ values in `context_configs.json`
- Update `fitness_computer.py` to use the per-cluster λ when available

3f. Add a `tune` subcommand to the oracle_builder CLI:
```
python -m thalamus.oracle_builder tune --oracle-dir /oracle
```
Runs all tuners and reports results without rebuilding the full oracle.

**Verification:** run tuning on a set of synthetic logs, confirm C/threshold values differ from defaults, confirm K differs from 20 for a non-default cluster count, confirm per-cluster λ values are stored.

---

### Step 4 — Semantic Scoring (Replace Lexical Metrics)

**Addresses gaps:** 1, 2, 3

**Why fourth:** with evaluation infrastructure in place, we can measure whether switching from lexical to semantic metrics actually improves classifier quality. We need Step 2 before Step 4 or we cannot verify improvement.

**What to build:**

4a. Add **BERTScore** as an optional scoring metric in `component_scoring/shared/metrics/`:
- `metric_bert_score.py`
- Uses `bert_score` package (lazy import; falls back gracefully if not installed)
- Computes F1 under BERTScore (model: `roberta-large` by default, configurable)
- Registered in `metrics_list.py` as `"bert_score"` (optional)

4b. Add an **LLM judge metric** in `component_scoring/shared/metrics/`:
- `metric_llm_judge.py`
- Prompt: given expected answer and candidate output, rate semantic correctness on 1–10
- Normalized to [0, 1]
- Requires `--judge-model` and `--judge-api-key` flags passed down from the CLI
- Registered as `"llm_judge"` (optional)

4c. Add a **metric selector** to the scoring CLI:
- `--metrics` flag accepting a comma-separated list: `f1,bigram_f1,bag_of_words,length_ratio,bert_score,llm_judge`
- Default remains the existing four lexical metrics for backwards compatibility
- `mean_score` computed over whichever metrics are selected

4d. Store the metric configuration in `scoring_matrix_*.json` under a `"metrics_used"` field so Stage 3 can detect mismatches between different scoring runs.

4e. Add a **combination scoring mode** (addresses gap 3):
- `--eval-combination-size N` flag on the `build` command
- When N > 1: during Stage 1 evaluation, assemble N-component combinations (random sampling from the full component list) and evaluate the query against the combination in context
- Contribution of each component to the combination's score is estimated by comparing the combination score to the score without that component (leave-one-out within the combination)
- This is optional and slow (N × LLM calls per combination); default remains single-component evaluation

**Verification:** build scoring matrices with `--metrics bert_score` and compare `mean_score` distribution to the lexical baseline. Confirm that paraphrase-equivalent outputs score higher with BERTScore than with F1.

---

### Step 5 — Drift Detection and Automatic Retraining Triggers

**Addresses gaps:** 19, 20, 21, 22

**Why fifth:** with better labels (Step 1) and measurement infrastructure (Step 2), we can now monitor whether the system is degrading.

**What to build:**

5a. Add a **query distribution monitor** in `shared/`:
- `distribution_monitor.py`
- Maintains a rolling embedding of recent queries (1-week and 4-week windows)
- Computes Jensen-Shannon divergence between the two windows on TF-IDF histograms
- Threshold: JS divergence > 0.15 → flag potential drift
- Writes `drift_status.json` to the oracle directory: `{"detected": bool, "js_divergence": float, "window_1w": int, "window_4w": int, "checked_at": timestamp}`

5b. Add an **oracle staleness checker** in `oracle_builder/`:
- `staleness_checker.py`
- Loads `context_configs.json` and the current component registry (re-scans component sources)
- Detects: new components not in the oracle, removed components still in the oracle, components with changed fingerprints
- Writes `staleness_report.json`: `{"stale": bool, "new_components": [...], "removed": [...], "changed": [...]}`

5c. Add a `status` subcommand to the oracle_builder CLI:
```
python -m thalamus.oracle_builder status --oracle-dir /oracle --skills-dir /skills ...
```
Runs drift monitor and staleness checker, prints a human-readable summary.

5d. Add a **retraining scheduler** in `oracle_builder/`:
- `retraining_scheduler.py`
- `should_retrain_classifier(oracle_dir) → bool`: returns True if any of:
  - Drift detected (from `drift_status.json`)
  - Turns accumulated since last training > configured threshold (default 500)
  - Last training date older than configured interval (default 7 days)
- `should_rebuild_oracle(oracle_dir) → bool`: returns True if any of:
  - Oracle is stale (from `staleness_report.json`)
  - Drift detected and last rebuild > 14 days ago

5e. Add a `check-rebuild` subcommand that prints a recommendation:
```
python -m thalamus.oracle_builder check-rebuild --oracle-dir /oracle
```
Prints: "REBUILD RECOMMENDED: [reasons]" or "NO REBUILD NEEDED".

5f. Add `--auto-threshold` flag to `train-classifier`:
- Before training, calls `retraining_scheduler.should_retrain_classifier()`
- If False: skips training and exits with status 0 and a message
- If True: proceeds with training

**Verification:** inject synthetic turn logs with a shifted query distribution; confirm `drift_status.json` reports `detected: true`. Add a new component to the source; confirm `staleness_report.json` reports it.

---

### Step 6 — Cross-Path Learning

**Addresses gaps:** 17, 18

**Why sixth:** requires the classifier infrastructure from Steps 2 and 3 (to load and evaluate classifiers) and the oracle infrastructure from Step 5 (to know what's in the oracle). Cannot be done without both paths fully operational.

**What to build:**

6a. Add **oracle-based classifier warm start** in `oracle_builder/classifier/component_classifier_trainer.py`:
- After building the training dataset but before fitting, check if `context_configs.json` exists
- If it exists: extract per-component relevance scores (mean cosine similarity across all cluster centroids × component centroid) as a weight vector
- Initialize logistic regression weights proportionally: `W_i ← scale × relevance_i` where scale is fit to the embedding dimension
- Pass initial weights to scikit-learn via `warm_start=True` and manual weight initialization
- Store `"initialized_from_oracle": true` in the classifier registry entry

6b. Add **classifier-informed fitness** to the GA in `oracle_builder/evolutionary/evolution/fitness_computer.py`:
- At build time, check if `classifier.pkl` (current version) exists
- If it exists and has validation F1 > 0.6 (configurable threshold): load it
- Add a third term to the fitness formula:
  ```
  fitness(b) = Σᵢ[mean_score_i × cosine(v, centroid_i) × bit_i]
               − λ_k × tokens/budget
               + α × Σᵢ[classifier_prob(cluster_centroid, i) × bit_i]
  ```
  where `classifier_prob` is the classifier's predicted inclusion probability for component i given the cluster centroid embedding, and `α` is a mixing weight (default 0.2)
- Add `--classifier-alpha` CLI flag to control α (default 0.2; 0.0 disables)
- Store `"classifier_informed": true` and `"classifier_alpha"` in `context_configs.json` metadata

6c. Add unit tests for both:
- Warm start: confirm weight magnitudes differ from random initialization in the direction of relevance scores
- Classifier-informed fitness: confirm fitness values differ between `alpha=0` and `alpha=0.2` on a synthetic dataset

**Verification:** build oracle with and without `--classifier-alpha 0.2` after a classifier is trained; confirm the `context_configs.json` metadata records the difference. Train classifier with and without warm start; compare training convergence speed.

---

### Step 7 — Learned Fitness Function

**Addresses gaps:** 5, 6

**Why seventh:** requires real logged data with good labels (Step 1), evaluation infrastructure (Step 2), cross-path signal (Step 6), and sufficient deployment maturity. Cannot meaningfully learn the fitness function without all of the above.

**What to build:**

7a. Add a **fitness dataset builder** in `oracle_builder/evolutionary/`:
- `fitness_dataset_builder.py`
- For each logged turn: extract (cluster_id, included_components_bitmask, outcome_quality)
- Map cluster_id from the turn's query embedding using the saved clusterer
- Features: component bitmask (N bits) + cluster_id one-hot (K bits) = N+K features
- Label: outcome_quality scalar
- Writes `fitness_training_data.npz` to oracle directory

7b. Add a **fitness model trainer** in `oracle_builder/evolutionary/`:
- `fitness_model_trainer.py`
- Trains a gradient boosting regressor (scikit-learn `GradientBoostingRegressor`) on the fitness dataset
- Minimum data guard: at least 200 turns required (otherwise returns None — no model trained)
- Saves `fitness_model.pkl` to oracle directory
- Evaluates on held-out 20% split (RMSE, R²)
- Applies the same promotion gate as the classifier: only promotes if R² improves

7c. Update `fitness_computer.py` to load and use `fitness_model.pkl` when available:
- If model exists: use it as the primary fitness signal
- Formula when model exists:
  ```
  fitness(b) = fitness_model.predict([cluster_onehot, bitmask]) − 0.1 × tokens/budget
  ```
  (token penalty retained; learned model replaces the quality × relevance sum)
- If model does not exist: fall back to existing hand-crafted formula unchanged
- Add `--disable-learned-fitness` flag to force the hand-crafted formula even when a model exists

7d. Add `train-fitness-model` subcommand to oracle_builder CLI:
```
python -m thalamus.oracle_builder train-fitness-model --oracle-dir /oracle --min-turns 200
```

**Verification:** on a synthetic dataset with a known component pair (A+B jointly useful), confirm the learned fitness model scores the bitmask [A=1, B=1] higher than [A=1, B=0] or [A=0, B=1]. Confirm the hand-crafted formula does not.

---

### Step 8 — Joint Component Modeling (Multi-Label Classifier)

**Addresses gaps:** 7, 8

**Why last:** requires abundant data (Steps 1–3 foundations, at least 500 turns per component type), a working evaluation framework, and the learned fitness model (Step 7) to verify that joint modeling actually captures combination effects the hand-crafted formula missed. This is the most architecturally complex change.

**What to build:**

8a. Add a **multi-label classifier** in `shared/`:
- `multi_label_classifier.py`
- Architecture: single model with N binary outputs sharing one input representation
- Implementation options (in order of complexity):
  - Option A (default): `MultiOutputClassifier` wrapping a shared `LogisticRegression` on a shared feature space — same W matrix but trained jointly with shared regularization
  - Option B (advanced): small MLP (input → 128 hidden → N outputs) via scikit-learn `MLPClassifier`
- Input: query embedding (d dimensions)
- Output: N probabilities, one per component

8b. Add a **joint trainer** in `oracle_builder/classifier/`:
- `joint_classifier_trainer.py`
- Builds a single feature matrix X (turns × d) and label matrix Y (turns × N)
- Y[turn, i] = 1 if component i was in context AND outcome quality > 0.5
- Trains the multi-label model using `MultiOutputClassifier(LogisticRegression(...))`
- Minimum data guard: at least 50 × N turns required (N = number of components)
- Saves as `joint_classifier.pkl` (distinct from `classifier.pkl` to allow parallel operation)
- Applies the same versioning and promotion gate from Step 2

8c. Add `JointClassifierSelector` in `context_selectors/`:
- `joint_classifier_selector.py`
- Accepts query embedding; returns N probabilities from the joint model
- Falls back to `ClassifierSelector` (independent models) if `joint_classifier.pkl` does not exist

8d. Add `train-joint-classifier` subcommand to the oracle_builder CLI:
```
python -m thalamus.oracle_builder train-joint-classifier --oracle-dir /oracle --min-turns 1000
```

8e. A/B comparison utility:
- After both `classifier.pkl` and `joint_classifier.pkl` exist, add `compare-classifiers` subcommand
- Runs both on the held-out validation set
- Prints side-by-side precision, recall, F1, AUC
- Prints joint model's macro F1 minus independent model's macro F1 (positive = joint model is better)

**Verification:** on a synthetic dataset with a known joint-necessity pair, confirm the joint model assigns high probability to both components together and low probability to either alone. Confirm the independent model assigns low probability to both.

---

## Summary Table

| Step | Addresses Gaps | Depends On |
|------|---------------|------------|
| 1 — Better outcome labels | 9, 10, 11, 12 | Nothing (can start immediately) |
| 2 — Evaluation + versioning | 13, 14, 15 | Step 1 (labels needed to measure) |
| 3 — Hyperparameter tuning | 16, 23, 24, 25, 26 | Step 2 (need measurement to tune) |
| 4 — Semantic scoring | 1, 2, 3 | Step 2 (need measurement to verify improvement) |
| 5 — Drift detection + retraining | 19, 20, 21, 22 | Step 2 (need oracle registry for staleness) |
| 6 — Cross-path learning | 17, 18 | Steps 2, 3, 5 (both paths must be mature) |
| 7 — Learned fitness function | 5, 6 | Step 6 (needs real data + cross-path signal) |
| 8 — Joint component modeling | 7, 8 | Steps 1–7 (needs data, evaluation, everything) |

All existing functionality (Path A Stages 1–5, Path B classifier, both selectors, shared utilities, CLI) is preserved unchanged. Each step adds new capability alongside what exists. No step removes or replaces existing code — it either extends it or adds alongside it with feature flags.
