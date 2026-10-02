# CLAUDE CODE IMPLEMENTATION PROMPT — Project TraceGRPO

**You are Claude Code running in a GPU VM. This document is your complete mission specification. Read it fully before writing any code. Work phase by phase. Do not skip acceptance gates.**

---

## 0. MISSION

Implement and experimentally validate **TraceGRPO**: a research system that trains a small open-source code LLM (Qwen3-4B primary, Qwen2.5-Coder-7B secondary) for **joint software vulnerability analysis** — binary detection, CWE typing, statement-level localization, and source→sink trace explanation — using **program-analysis-verifiable reinforcement rewards** (patch-line IoU and DFG/call-graph trace fidelity computed with Joern) instead of LLM-judge rewards, evaluated under **random, chronological, and repository-disjoint splits** plus a post-cutoff out-of-distribution (OOD) CVE set.

This implements the research proposal in the companion document `AI_Vulnerability_Detection_Literature_Review_and_Research_Proposal.md` (sections 18–29). The base paper being extended is **VULPO** (arXiv:2511.11896). Read sections 12–14 (base paper architecture/results/limitations), 19 (proposed architecture), 21–25 (methodology/metrics/ablations/generalization/error analysis) of that report if it is present in the workspace; otherwise this prompt is self-contained.

**Headline hypotheses to test:**
- **H1:** Verifiable-reward training yields ≥5 absolute points improvement in statement-level F1 and ≥3 points in P-Pass@1 over judge-reward RL (VULPO replication) at matched backbone/data/budget, 95% CIs excluding 0 across ≥5 seeds.
- **H2:** Under repository-disjoint splits, TraceGRPO's F1 degradation is ≥30% smaller (relative) than SFT-only and judge-RL baselines'.
- **H3:** Trace validity (DFG-realizable claimed source→sink paths) improves ≥15 absolute points; label-correct/reason-wrong rate drops.

---

## 1. GROUND RULES (non-negotiable)

1. **Never fabricate data, results, or metrics.** If an experiment cannot run, say so and log why. Every number in `results/` must be produced by an executed script with a logged seed and config hash.
2. **Commit early, commit often.** Git-init the repo immediately. Commit after every working module. Tag each phase completion (`phase-1-done`, etc.).
3. **Smoke-test before scale.** Every pipeline stage must run end-to-end on 50 samples before running on the full corpus. Never launch a multi-hour training run that has not passed a 10-minute smoke run.
4. **Determinism & logging.** Every script takes `--seed`. Log: git hash, config YAML, package versions (`pip freeze`), GPU model, start/end time. Use Weights & Biases if available, else local JSONL logs + CSV summaries.
5. **Checkpoint everything.** Training checkpoints every N steps to disk; preprocessing writes intermediate artifacts so re-runs resume rather than restart.
6. **Fail loudly.** Assert on schema violations (missing fields, parse failures above threshold, empty rewards). Never silently drop >2% of samples without logging a warning with counts and reasons.
7. **No test-set leakage into training decisions.** Model/prompt/hyperparameter selection uses the validation split only. The test split and the OOD set are touched only by the final evaluation harness.
8. **Ask before destructive or expensive actions.** If a step will take >6 GPU-hours or delete data, stop and ask.
9. **Keep the paper trail.** Every design decision that deviates from this prompt goes into `DECISIONS.md` with date and rationale.
10. **Structure of work:** complete phases in order; within a phase, write tests first where practical (pytest).

---

## 2. ENVIRONMENT SETUP (Phase 0)

### 2.1 Assumed hardware

Single GPU VM, ideally 1×A100/H100 80GB or 2×48GB (A6000/L40S). Minimum viable: 1×48GB with QLoRA. ≥16 CPU cores, ≥64GB RAM, ≥500GB free disk (CPGs + datasets + checkpoints).

First action: run `nvidia-smi`, `python3 --version`, `df -h`, `nproc`, `free -g` and record into `ENVIRONMENT.md`. If VRAM < 40GB, switch all configs to the QLoRA fallback path (§10.4) and note it.

### 2.2 Software stack

```bash
# System
sudo apt-get update && sudo apt-get install -y openjdk-17-jdk unzip git-lfs build-essential
# Python 3.10+ env
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.* accelerate datasets peft trl==0.* vllm==0.* \
            deepspeed flash-attn --no-build-isolation
pip install tree-sitter tree-sitter-c tree-sitter-cpp pygments unidiff \
            pandas numpy scipy scikit-learn matplotlib seaborn \
            pytest pyyaml tqdm jsonlines networkx pydantic
```

**Joern** (critical dependency — the CPG engine):
```bash
wget https://github.com/joernio/joern/releases/latest/download/joern-install.sh
chmod +x joern-install.sh && ./joern-install.sh --version=v2.*  # pin the actual latest stable
joern --version  # verify
```
If Joern installation fails, fall back to `joern-cli` docker image OR to a pure-Python pipeline using `tree-sitter` (AST) + `joern-parse` alternatives; document any fallback in `DECISIONS.md` — reward semantics must remain identical.

**Models** (Hugging Face): `Qwen/Qwen3-4B` (primary), `Qwen/Qwen2.5-Coder-7B-Instruct` (secondary). Teacher for cold-start traces: use the strongest available open model you can serve locally via vLLM (DeepSeek-R1-Distill-Qwen-32B if VRAM allows on a second GPU or quantized; otherwise `Qwen2.5-Coder-32B-Instruct-AWQ`; document the choice). If no large teacher fits, use multi-round self-generation from the 7B instruct model with best-of-8 metadata-filtered sampling (this is a documented fallback, not a blocker).

### 2.3 Repo layout (create exactly this)

```
tracegrpo/
├── README.md                  # project overview + how to reproduce
├── DECISIONS.md               # dated design decisions
├── ENVIRONMENT.md             # hardware/software record
├── configs/                   # all experiment configs (YAML)
│   ├── data/  sft/  rl/  eval/
├── data/
│   ├── raw/                   # downloaded datasets (never modify)
│   ├── processed/             # corpus parquet/jsonl with all annotations
│   └── splits/                # split manifests (sample_id lists) — versioned
├── src/
│   ├── data/                  # loaders, normalizers, split builders
│   ├── cpg/                   # joern wrappers, context extractor, statement mapper
│   ├── rewards/               # R1..R5 reward functions + trace checker
│   ├── contracts/             # output contract schema + parser
│   ├── sft/                   # cold-start SFT pipeline
│   ├── rl/                    # GRPO training loop (TRL) + custom reward server
│   ├── eval/                  # metrics, harness, statistical tests
│   ├── baselines/             # zeroshot, judge-RL replication, encoders
│   └── utils/                 # seeding, logging, io, gpu helpers
├── scripts/                   # thin CLI entrypoints (one per pipeline stage)
├── tests/                     # pytest; includes reward-unit tests
├── notebooks/                 # exploration only; nothing here is authoritative
├── results/                   # all runs: results/<run_name>/{metrics.json, logs, preds.jsonl}
└── checkpoints/
```

`pip install -e .` with a `pyproject.toml`. All scripts accept `--config` and `--seed`.

### 2.4 Phase 0 acceptance gate

- [ ] `pytest` runs (even if only trivial tests) green.
- [ ] `joern --version` works; a 20-line C file parses to a CPG and a test query returns nodes.
- [ ] vLLM serves Qwen3-4B; a hello-world generation completes; peak VRAM recorded.
- [ ] `ENVIRONMENT.md` complete; repo committed and pushed (if remote available).

---

## 3. PHASE 1 — DATA ACQUISITION & SPLIT CONSTRUCTION (weeks 1–2)

### 3.1 Acquire datasets into `data/raw/` (record exact versions/hashes)

1. **PrimeVul** — clone `https://github.com/DLVulDet/PrimeVul`. Files of interest: `primevul_train.jsonl`, `primevul_valid.jsonl`, `primevul_test.jsonl`, and the **paired** files `primevul_test_paired.jsonl`. Fields include `idx`, `project`, `commit_id`, `func`, `target` (1=vulnerable), `cwe`, `cve`, `func_before`/patch info. Verify counts: 6,968 vulnerable / 228,800 benign total.
2. **SecVulEval** — locate the official release (arXiv:2505.19828; check the authors' GitHub/HF). If the dataset is not publicly downloadable, **do not substitute silently** — record in `DECISIONS.md` and construct the statement-level annotations yourself on PrimeVul (§4.3 does this anyway) and use SecVulEval only if obtained.
3. **MegaVul** — `https://github.com/Icyrockton/MegaVul` (or its HF mirror). Needed for the inter-procedural stretch subset and additional CVE metadata.
4. **Post-cutoff OOD set (≥200 pairs)** — collect CVEs with C/C++ patches disclosed **after the backbone's training cutoff** (for Qwen3-4B, treat cutoff as ~mid-2024; to be safe use CVEs from **2025-01-01 onward**). Source: GitHub Security Advisories + NVD + CVEfixes-style commit mining. Pipeline: query GitHub API for advisories with fix commits in C/C++ repos → clone repo at parent commit → extract changed function(s) (vulnerable and patched versions) → record metadata (CVE id, CWE, dates, project, commit). Target ≥200 usable single-function-change pairs; keep a `collection_log.jsonl` with rejection reasons. **Contamination screen:** for each candidate, prompt the backbone model with the CVE id and ask it to describe the bug; log responses; flag (do not auto-drop) samples the model clearly memorizes.

### 3.2 Normalize into a unified sample schema (`data/processed/corpus.parquet`)

```
sample_id        str   # f"{source}:{cve}:{commit}:{func_hash}"
source           str   # primevul | megavul | ood
project          str
cve, cwe         str (nullable)
commit_id        str
func_vuln        str   # vulnerable version source
func_patch       str   # patched version source
diff             str   # unified diff of the function
patch_lines_vuln list[int]  # 1-based line numbers in func_vuln removed/changed by patch
commit_date      datetime
target           int   # 1 = vulnerable function (func_vuln), 0 = benign (func_patch)
pair_id          str   # links vuln/patch halves
```

Write a validator (`tests/test_corpus_schema.py`): no null `func_vuln`, `patch_lines_vuln` non-empty for target=1, dates parseable, pair integrity (every pair_id has exactly 2 rows).

### 3.3 Build splits (`data/splits/`) — three regimes, each with train/val/test sample_id lists

1. **Random split** (80/10/10, stratified by target) — for continuity with prior work; flagged as inflated.
2. **Chronological split** — train on commits before 2022-01-01, validate 2022, test 2023+ (adjust cutoffs so val/test each hold ≥1,500 samples; document actuals).
3. **Repository-disjoint split** — all samples of a project assigned to exactly one partition (greedy balanced assignment by vulnerable-sample count; 5 folds stored for cross-validation).
4. **Pair integrity constraint everywhere:** a `pair_id` never crosses partitions (vuln and patched halves together).
5. **De-duplication:** before splitting, dedup on normalized function text (strip comments/whitespace, hash) with near-duplicate detection via 5-gram Jaccard ≥ 0.9 (MinHash/LSH). Log how many samples were removed.

Write `scripts/build_splits.py` + `tests/test_splits.py` asserting: disjointness, pair integrity, no near-dup across partitions (spot-check), expected sizes.

### 3.4 Phase 1 acceptance gate

- [ ] Corpus validates; counts and class balance printed and saved to `results/phase1/data_summary.json`.
- [ ] Three split manifests + OOD set built and versioned; leakage tests green.
- [ ] A data card (`data/DATA_CARD.md`): sources, sizes, CWE distribution plot, date histogram, dedup stats.

---

## 4. PHASE 2 — STRUCTURAL ANNOTATION PIPELINE (weeks 2–5)

This is the highest-risk engineering phase. Budget accordingly. Everything here must be **CPU-parallelizable** (multiprocessing; target 16 workers) and **resumable** (per-sample artifact files, skip if exists).

### 4.1 CPG construction (`src/cpg/joern_runner.py`)

For each unique `(project, commit_id)` needing context: we do **not** clone full repos at scale for the main corpus (that is the stretch goal). For the main pipeline, parse at **function level with synthesized compilation units**: wrap `func_vuln` in a minimal translation unit with stub declarations for unresolved identifiers (this is how function-level CPG tools operate); run Joern's `c2cpg` → load into Joern → export AST/CFG/DFG subgraph for the function via a Scala/py4j bridge or `joern-export`. Cache CPGs on disk (`data/cpg_cache/<hash>.dot` / graphSON).

For a **calibration subset of 200 samples**, additionally clone the full repo at the vulnerable commit, build a full CPG, and compare function-level DFG edges vs repo-level DFG edges for the target function; report edge-recall of the lightweight approach in `results/phase2/cpg_calibration.json`. This calibrates reward-checker reliability (threat to validity #internal).

### 4.2 Statement mapping (`src/cpg/statement_mapper.py`)

Map each entry of `patch_lines_vuln` (diff line numbers) to **statement nodes** in the AST:

1. Parse `func_vuln` with tree-sitter C/C++.
2. Map line → smallest AST node whose span covers the line and whose type ∈ statement set (expression_statement, declaration, if_statement, for_statement, while_statement, return_statement, call_expression, assignment_expression, etc. — maintain an explicit allowlist in `configs/cpg/statement_types.yml`).
3. Output per sample: `L1` = set of statement node ids + their line ranges (the patch statements).

Then compute the **graded localization targets**:
- `L1`: exact patch statements.
- `L2`: L1 ∪ statements in the **backward def-use slice** from L1 within the function (using the DFG: statements that define variables used in L1, transitively up to depth 3, plus control-dependency parents) — computed from the Joern DFG where available, else from a tree-sitter-based intra-function def-use analysis (`src/cpg/defuse.py`; implement reaching-definitions-lite).
- `L3`: the whole function body (degenerate upper bound used only for reward-guard calibration).

### 4.3 Trace ground truth (`src/cpg/trace_miner.py`)

For vulnerable samples, mine candidate **source→sink traces** ending at each L1 statement:

- **Sink**: the L1 statement (or its dominant dangerous call if recognizable — maintain a sink-pattern table per CWE in `configs/cpg/sink_patterns.yml`, seeded with ~150 patterns from known CWE rules: e.g., memcpy/strcpy/sprintf for CWE-119/120, free() for CWE-415/416, etc.).
- **Source**: statements defining variables that flow into the sink (backward DFG reachability), prioritizing parameters, return-of-call assignments, and global reads.
- Output: up to 3 shortest DFG paths per sample: `trace_gt = [[stmt_id, ...], ...]` with per-step edge types (def-use / call / control).
- Samples with no extractable trace are flagged `trace_unverifiable=True` — **kept for evaluation but excluded from R3 reward computation** (mirrors mono's undecidable-patch finding).

### 4.4 Reward-checker core (`src/rewards/checker.py`)

Pure functions, fully unit-tested:

- `statement_iou(pred_lines: set[int], gold: set[int]) -> float` — |∩|/|∪| over line sets (operate on line numbers mapped from statement ids; this makes R2 robust to formatting).
- `trace_validity(pred_trace: list[int], dfg: Graph) -> tuple[bool, float]` — each consecutive pair must be a real DFG/call edge; first node must be a source (per §4.3 definition), last must reach an L1 sink. Returns (fully_valid, prefix_fraction). Prefix fraction = longest valid prefix / total claimed steps.
- `is_degenerate_localization(pred_lines, func_len) -> bool` — True if coverage > 60% of function lines or empty.

**Unit tests (mandatory, ≥40 cases):** hand-crafted C functions with known patch lines and known def-use paths, including tricky cases (macros, multi-line statements, nested calls, shadowed variables). Trace checker must reach ≥95% agreement with your hand labels on these before Phase 5.

### 4.5 Context extraction (VULPO-style, `src/cpg/context_extractor.py`)

For each target function, extract **lightweight repository context**: when the sample's source dataset provides repo access (or when we clone for the calibration/OOD subsets), pull (a) callee function bodies (1 hop) referenced in the function, (b) callee signatures only (2 hops), (c) relevant global/type declarations. Budget: ≤4,096 tokens of context per sample (truncate callee bodies first, then drop 2-hop signatures). Where repo access is unavailable, context = empty string and `has_context=False` — never fabricate context.

### 4.6 Phase 2 acceptance gate

- [ ] ≥90% of corpus samples have L1; ≥70% have L2; ≥50% have ≥1 trace_gt (log exact rates; shortfalls with reasons in `results/phase2/coverage.json`).
- [ ] Reward checker unit tests green; calibration report written.
- [ ] 20 randomly sampled annotations manually eyeballed and recorded in `results/phase2/manual_audit.md`.

---

## 5. PHASE 3 — OUTPUT CONTRACT & PROMPTING (week 5)

### 5.1 The joint output contract (`src/contracts/schema.py`)

The model must emit exactly one structured block:

```
<analysis>
<label>VULNERABLE|SAFE</label>
<cwe>CWE-###|NONE</cwe>
<statements>comma-separated line ranges, e.g. 12-14, 27</statements>
<trace>
  <step>line 12: buf = getenv("PATH")   [source: env input]</step>
  <step>line 27: memcpy(dst, buf, n)    [sink: unchecked copy]</step>
</trace>
<explanation>2–5 sentences: root cause, propagation, security consequence</explanation>
</analysis>
```

Implement a strict parser (`src/contracts/parser.py`) returning a typed object; parse failure → `format_ok=False` (feeds R5). Parser must be fuzz-tested (≥100 malformed inputs; never crash).

### 5.2 Prompt template (`src/contracts/prompt.py`)

System prompt: security analyst role; instructions to ground every claim in provided code/context; explicit output contract; instruction to emit line numbers from the shown numbered code. User prompt: numbered `func_vuln`, optional context block (§4.5), CVE/CWE metadata **only in teacher-generation mode** (never at train/eval inference — that would leak labels). Keep total input ≤28k tokens for the 32k window.

### 5.3 Phase 3 gate

- [ ] Round-trip: generate → parse → re-serialize works on 50 smoke samples with 100% parse success after 1 retry-with-repair instruction.

---

## 6. PHASE 4 — COLD-START SFT (weeks 5–7)

### 6.1 Teacher trace generation (`src/sft/gen_teacher.py`)

On the **train split only** (chronological split is the master split; reuse it for all training):

1. Serve the teacher with vLLM.
2. For each train sample, generate **8 reasoning rollouts** with the metadata-informed prompt (teacher sees CVE description + CWE + patch diff — this is allowed for supervision generation).
3. **Filter** (metadata agreement, per VULPO recipe): keep traces where `<label>` matches ground truth AND `<cwe>` matches (or is a parent category) AND statement overlap with L1∪L2 IoU ≥ 0.3. Target retention ≥35%; if lower, raise temperature diversity, not thresholds.
4. Cap at 1–2 traces per sample; dedupe near-identical traces.

### 6.2 SFT run (`src/sft/train_sft.py`)

- Qwen3-4B, LoRA r=64, alpha=128, dropout 0.05, bf16, lr 1e-4 cosine, warmup 3%, **2 epochs** (literature: 1–3 optimal; 5 hurts downstream RL — ablate later), global batch 64 (grad accumulation), max len 8k (pack traces + code), masking loss on the prompt.
- Eval every 500 steps on val: label accuracy + parse rate + mean statement IoU of parsed outputs (soft signal only).
- Save `checkpoints/sft-final`. **Smoke rule:** first do a 200-sample, 50-step run and verify loss decreases and generations parse ≥80%.

### 6.3 Phase 4 gate

- [ ] Val parse rate ≥95%; val label accuracy meaningfully above majority baseline; artifacts + logs saved.

---

## 7. PHASE 5 — VERIFIABLE REWARDS (week 7)

Implement `src/rewards/` as pure functions over (parsed_output, sample_annotations, cpg_cache). **No LLM calls inside R1/R2/R3/R5.**

```
R1 outcome:      +1 if label correct else -1                      (gate)
R2 localization: IoU(pred_statement_lines, L1 ∪ L2)  ∈ [0,1]      (only if label correct)
R3 trace:        0.5*fully_valid + 0.5*prefix_fraction ∈ [0,1]    (only if label correct;
                 skipped — not zeroed — when trace_unverifiable)
R4 reasoning:    auxiliary LLM-judge score ∈ {0, 0.5, 1}, weight ≤ 0.2 total,
                 judge = Qwen2.5-Coder-7B-Instruct (small, fixed), ablatable
R5 guards:       -0.5 if format_ok == False
                 -0.5 if is_degenerate_localization(...)
                 -0.5 if statements non-empty while label == SAFE (consistency)
Total = R1 * (1 + R2 + R3 + 0.2*R4) + R5   # gated multiplicative form
```

Also implement `w_l` (label-level difficulty weight: vulnerable samples weighted >1; default 1.5, configurable) and `w_s` (sample-level: inversely proportional to within-group correctness rate) per VULPO's difficulty-adaptive scaling.

**Tests:** golden reward vectors on the Phase 4.2 hand-crafted fixtures; property tests (rewards bounded; R2 monotonic in overlap; R3 monotonic in prefix).

---

## 8. PHASE 6 — GRPO TRAINING (weeks 8–11)

### 8.1 Implementation

Use TRL's `GRPOTrainer` (or verl if TRL lacks hooks) with a **custom reward function** that looks up precomputed annotations and calls the reward suite; run rollouts with vLLM (`use_vllm=True`, colocate mode). Config (`configs/rl/grpo_4b.yaml`):

```
model: checkpoints/sft-final
num_generations: 8
max_completion_length: 1536
max_prompt_length: 28672        # 32k total window
temperature: 1.0
beta (KL): 0.0                  # group-mean advantage, no std norm, per VULPO
lr: 1e-6, cosine, warmup 10
per_device_train_batch_size: 1  # with grad accum to effective 64 prompts/step
gradient_checkpointing: true
bf16: true
lora: r=64 (continue from SFT adapter or merge+new adapter — decide and document)
save_steps: 100
logging_steps: 5
seeds: [42, 43, 44] minimum; [45, 46] for headline claims
```

### 8.2 Reward server performance

CPG lookups must be pre-cached; rewards must compute in <50ms/sample. Profile once; if slower, precompute serialized DFGs into the corpus parquet.

### 8.3 Monitoring

Track per step: mean reward decomposed (R1..R5), parse rate, degenerate-localization rate, KL, completion length (watch for collapse/blow-up), val P-Pass@1 every 200 steps. **Early stop** if parse rate <90% for 3 consecutive evals or degenerate localization >10%.

### 8.4 QLoRA fallback (if VRAM < 40GB)

4-bit NF4 base + LoRA r=32, `max_completion_length=1024`, prompt budget 16k, num_generations 6. Document expected quality delta and prefer renting a bigger instance if possible.

### 8.5 Phase 6 gate

- [ ] Training curves healthy; best-val checkpoint selected on val P-Pass@1 only.
- [ ] `results/rl_main/` contains metrics.json, preds.jsonl (val), full logs, config, seeds.

---

## 9. PHASE 7 — BASELINES (weeks 9–12, parallel with Phase 6 where GPU allows)

All baselines on **identical splits and identical evaluation harness**:

1. **Zero-shot backbone** (Qwen3-4B) with the Phase 3 prompt.
2. **SFT-only** (Phase 6.2 checkpoint).
3. **Judge-RL replication of VULPO** — same GRPO config, but R2/R3 replaced by judge scores from the R4-style judge prompted with patch diff + CVE description (per VULPO §reward design). This is the critical baseline; implement carefully and note it is a re-implementation from the paper spec (their artifact release is unverified).
4. **Outcome-only GRPO** (R1 + R5 only).
5. **LineVul-style encoder baseline**: fine-tune CodeBERT/UniXcoder for binary detection (function level) — reuse HuggingFace `Trainer`; report detection metrics only.
6. **Prompting reference**: Qwen2.5-Coder-32B-Instruct (AWQ) zero-shot with the same prompt — measures the inference-time-scaffolding frontier.
7. **(Stretch)** VulTriage-style triple-path prompt augmentation on the 32B model.

Each baseline: `results/baselines/<name>/{metrics.json, preds.jsonl}`.

---

## 10. PHASE 8 — EVALUATION HARNESS (weeks 11–13)

### 10.1 Metrics (`src/eval/metrics.py`)

- **Detection:** accuracy, precision, recall, F1, MCC, PR-AUC.
- **Pair-wise (PrimeVul/VULPO-compatible):** P-C (both pair halves correct), P-V (both labeled vulnerable), P-B (both benign), P-R (reverse), **P-Pass@1 / P-Pass@8**; also **VP-S** (vulnerable-pair strict: correct label + correct CWE + statement IoU ≥ 0.5).
- **Localization:** statement precision/recall/F1 **with reasoning gate** (count correct only if trace_valid or explanation passes judge check — report both gated and ungated); Top-1/3/5 line accuracy; mean IoU.
- **Trace:** trace validity rate; mean prefix fraction; **label-correct/reason-wrong rate**.
- **CWE:** macro-F1, head (top-12 CWEs) vs tail split.
- **Efficiency:** tokens/sample (prompt+completion), latency, peak VRAM, judge-calls per train sample (TraceGRPO should be ≈0 vs judge-RL ≈2).
- **Statistics:** bootstrap 95% CIs (10k resamples) at the pair level; paired comparisons across seeds (Wilcoxon signed-rank + Cohen's d); Holm correction across RQ families.

### 10.2 Harness behavior

`scripts/run_eval.py --model <ckpt> --split <manifest> --seeds ...` writes `results/eval/<model>__<split>/`. All models evaluated through the **same** generation path (vLLM, temp 0 for pass@1, temp 0.8 n=8 for pass@8).

### 10.3 Evaluation battery (the full matrix)

| Model × Split | random | chronological | repo-disjoint | OOD (post-cutoff) | perturbation |
|---|---|---|---|---|---|
| zeroshot-4B | ✓ | ✓ | ✓ | ✓ | ✓ |
| SFT-only | ✓ | ✓ | ✓ | ✓ | ✓ |
| judge-RL (VULPO-rep) | ✓ | ✓ | ✓ | ✓ | ✓ |
| outcome-GRPO | ✓ | ✓ | ✓ | ✓ | – |
| **TraceGRPO** | ✓ | ✓ | ✓ | ✓ | ✓ |
| 32B-prompt ref | – | ✓ | ✓ | ✓ | – |
| encoder (LineVul) | ✓ | ✓ | ✓ | – | – |

**Perturbation eval** (`src/eval/perturb.py`): semantic-preserving rewrites of test functions — identifier renaming, dead-code insertion, equivalent refactoring (e.g., `for`↔`while`) — measure decision-flip rate and ΔF1 (Semantic-Trap protocol).

### 10.4 Phase 8 gate

- [ ] H1/H2/H3 tested with CIs and effect sizes; a `results/HEADLINE_RESULTS.md` table.
- [ ] Every number traceable to a preds.jsonl + config + seed.

---

## 11. PHASE 9 — ABLATIONS (weeks 13–15)

Each ablation = a config change on the chronological split, seeds {42,43,44}:

(a) −R2 · (b) −R3 · (c) −R2−R3 (=judge-only, should ≈ baseline 3) · (d) −R4 (fully verifiable) · (e) −repo context · (f) −w_l/w_s · (g) R2 target L1-only vs L1∪L2 · (h) reward-source swap (judge scores for R2/R3 instead of program analysis — isolates verifiability) · (i) backbone swap to Qwen2.5-Coder-7B · (j) SFT epochs {1,2,5} interaction.

Report each ablation across detection + localization + trace metrics; highlight where verifiable rewards specifically help/hurt.

---

## 12. PHASE 10 — ERROR ANALYSIS & HUMAN AUDIT (weeks 15–16)

`src/eval/error_analysis.py`: partition FP/FN by CWE (head/tail), project, function length quartile, cyclomatic complexity (radon/lizard), context requirement (intra-function vs needs-callee), data-flow distance (trace_gt length), training-frequency of CWE, trace-validity of emitted explanation. Produce plots into `results/error_analysis/`.

Human audit protocol: sample 150 model outputs (50 each from chronological/repo-disjoint/OOD); a human (the researcher) grades trace correctness; compute Cohen's κ vs the automatic checker. Store the grading sheet as CSV in `results/human_audit/`. Provide the sampling + instructions tooling; the actual grading is done by the human — **prepare everything, then stop and hand over**.

---

## 13. STRETCH GOALS (only after all gates pass)

1. **Inter-procedural subset:** mine MegaVul CVEs whose patches span multiple functions; build 2–3-hop context slices + cross-function traces; fine-tune/evaluate.
2. **VulnGym-style agent comparison:** run one open coding agent on the OOD set for a cost/quality comparison.

---

## 14. DELIVERABLES CHECKLIST (definition of done)

- [ ] Full repo, clean README with reproduction commands, tagged phases.
- [ ] Data card, split manifests, corpus parquet with all annotations.
- [ ] Reward suite with unit tests + calibration reports.
- [ ] SFT + GRPO training code with configs; logs for all runs.
- [ ] Evaluation harness + full results matrix + HEADLINE_RESULTS.md.
- [ ] Ablation table; error analysis plots; human-audit package.
- [ ] `PAPER_RESULTS_SUMMARY.md`: every table/figure the paper needs, with generation script references.
- [ ] H1/H2/H3 verdicts stated honestly, including negative results.

---

## 15. KEY IMPLEMENTATION NOTES & PITFALLS (read twice)

1. **Line-number grounding is the silent killer.** Always present code to the model with explicit 1-based line numbers, and require `<statements>` to reference those numbers. Never parse statement ids from raw text position heuristics at eval time.
2. **Pair leakage is easy to reintroduce** — the pair-integrity constraint must be enforced in the split builder AND asserted in tests AND checked again at train-load time.
3. **Reward hacking watchlist:** (i) predicting SAFE everywhere (covered by w_l and PR-AUC monitoring); (ii) emitting all-lines localization (R5 guard + IoU makes this self-defeating — verify empirically); (iii) trace padding with plausible steps (prefix_fraction caps gains); (iv) format collapse (parse-rate early stop).
4. **vLLM + TRL colocate memory:** if OOM during rollouts, reduce `max_prompt_length` first, then `num_generations`, then enable sleep mode; never silently truncate code mid-function — drop over-long samples with a logged counter.
5. **Joern flakiness:** wrap every Joern call with timeout + retry + fallback to tree-sitter def-use; cache aggressively; a 5–10% CPG failure rate is normal — keep rewards computable without Joern for those samples (mark partial).
6. **Judge discipline:** the R4/auxiliary judge and the VULPO-replication judge must be the **same fixed model and prompt** (temperature 0); store all judge outputs for audit.
7. **Chronological master split:** do all training on the chronological train split; random-split numbers are reported only for literature comparability.
8. **32k context:** verify the tokenizer length of every prompt; log a histogram; samples exceeding budget are truncated per §4.5 rules, never mid-statement.
9. **Reproducibility of "best checkpoint":** selection criterion (val P-Pass@1) is fixed in advance; no test-informed selection.
10. **If VULPO's official artifact becomes available** (check once at start: search the paper page for a GitHub link), prefer using their ContextVul directly but still compute YOUR annotations on top; document any differences.

---

## 16. SUGGESTED EXECUTION ORDER (summary)

```
Phase 0 env → Phase 1 data+splits → Phase 2 CPG/annotations/reward-checker (+tests)
→ Phase 3 contract → Phase 4 SFT → Phase 5 rewards → Phase 6 GRPO
→ Phase 7 baselines → Phase 8 evaluation battery → Phase 9 ablations
→ Phase 10 error analysis/audit → stretch goals → PAPER_RESULTS_SUMMARY.md
```

Estimated timeline: ~16–20 weeks on 1×80GB GPU + 16 CPU cores. Highest risk: Phase 2 (structural annotation). If Phase 2 coverage gates cannot be met after honest effort, **stop and report** rather than lowering quality silently — a documented negative result about annotation feasibility is itself valuable.

**Begin now with Phase 0. Print the environment report, set up the repo skeleton, and commit.**
