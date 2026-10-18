# Fine-Grained Error Typing Data for Process Reward Models

This repository contains the data-construction code and data from the data-construction part of a
team course project, *Process Reward Modeling in LLMs* (Heidelberg University, 2026), written by
Qin Yan. The goal is to build training data for a Process Reward Model (PRM) at four error-label
granularities, extending the two-category error taxonomy of PathFinder-PRM into a finer, four-category
one.

Key points of the approach:

- A four-category error taxonomy (Calculation, Method, Prior-step consistency, Domain consistency) that extends PathFinder-PRM's two-category design.
- LLM-based labeling of candidate steps from PRM800K, with filtering for consistency against PRM800K's human ratings.
- An extra re-confirmation pass for the rarest category (Prior-step consistency), re-labeling it twice more and keeping only labels that reproduce.
- A balanced 1:1 sample of error and correct/suboptimal steps, matched on context length and prior-step count.
- Exclusion of all questions that overlap with the PRMBench evaluation benchmark.
- Four training-data versions (0/1/2/4-class) that isolate the effect of label granularity.

The full method, including the taxonomy design, the annotation and filtering procedure, and the
dataset versions, is described in [`report/data_construction.pdf`](report/data_construction.pdf).

## Directory structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── .env.example
├── scripts/                  # pipeline scripts, run in the order listed below
├── dataset_v1/                # final deliverable: 4 training-format datasets + validation split
├── outputs/                   # intermediate files (frozen pilot sample included; the rest is
│                               # created at runtime, see below)
└── report/
    └── data_construction.pdf  # full write-up of the data-construction method
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# then edit .env and set DEEPSEEK_API_KEY=<your key>
# GEMINI_API_KEY is only needed if you run the Gemini-based pilot script (label_pilot_4class.py)
```

Scripts that call the DeepSeek or Gemini API load `.env` from this directory.

## PRM800K

This project uses the 4 PRM800K jsonl files (`phase1_train.jsonl`, `phase1_test.jsonl`, `phase2_train.jsonl`, `phase2_test.jsonl`), as released with [OpenAI's PRM800K repository](https://github.com/openai/prm800k).

Download them and place them in `PRM800K/`, or point `scripts/analyze_prm800k.py` at a different location by setting the `PRM800K_DIR` environment variable to an absolute or relative path.

Intermediate results (API outputs, candidate pools, recheck logs, etc.) are written to `outputs/`, created automatically on first run. Final training data is written to `dataset_v1/`.

`outputs/pilot_4class_sample_ds.json` is included: it is the frozen list of 200 pilot-stage labeled samples that `label_batch_4class_ds.py` excludes from its batch-labeling candidate pool, so the pilot sample is never labeled twice.

## How to run: scripts (in this order)

| # | Script | What it does |
|---|--------|---------------|
| 1 | `analyze_prm800k.py` | Loads and expands the 4 PRM800K jsonl files into per-step records; provides `load_prm800k_dataset`/`expand_records` used by the rest of the pipeline. |
| 2 | `prompt_4class_v3.py` | Defines the final four-way error-labeling prompt (`CLS_PROMPT_4CLASS`). |
| 3 | `label_batch_4class_ds.py` | Sends candidate steps (human rating = -1) to the DeepSeek API with the v3 prompt and writes per-batch labeling results. |
| 4 | `recheck_prior.py` | Re-labels the kept `prior_ok=0` records twice (thinking off / thinking on) to check reproducibility, for batches 1+2. |
| 5 | `recheck_prior_batch3.py` | Same re-labeling check as above, for batch 3 only (skipping triples already covered). |
| 6 | `build_confirmed_labels.py` | Confirms or flips `prior_ok` based on the recheck results and produces the confirmed label set. |
| 7 | `build_selection_v1.py` | Excludes PRMBench-overlapping questions, then selects a balanced sample of error and correct/suboptimal steps matched on context length and prior-step count. |
| 8 | `build_format_preview.py` | Builds the four training-format rows (0/1/2/4-class) for a given record; provides `build_all_versions` used by the final assembly script. |
| 9 | `build_dataset_v1.py` | Applies the row-construction logic in `build_format_preview.py` to the full selection and writes the four final jsonl files plus the validation-question-id split. |

### Helper modules

Not run directly; imported by the scripts above.

| Script | What it does |
|--------|---------------|
| `prompt_4class_v2.py` | Defines the earlier (v2) four-way error-labeling prompt, used by the pilot scripts below. |
| `label_pilot_4class.py` | Pilot labeling script using the Gemini API and the v2 prompt on a small stratified sample; also provides `enc` (the tiktoken tokenizer), `make_full_context`, and retry constants reused by later scripts. |
| `label_pilot_4class_ds.py` | DeepSeek variant of the pilot labeling script (same v2 prompt); provides the `RunConfig`/`RunState`/retry/backoff framework reused by `label_batch_4class_ds.py`, `recheck_prior.py`, and `recheck_prior_batch3.py`. |

Note: `build_selection_v1.py` also reads PRMBench's preview data to exclude overlapping questions, expected at `../PathFinder-PRM/src/evaluation/reward_guided_search/eval_data/PRMBench_Preview/` relative to this directory (i.e. a checkout of the [PathFinder-PRM](https://github.com/declare-lab/PathFinder-PRM) repository as a sibling of this one).

## Dataset: `dataset_v1/`

Four training-format jsonl files, all built from the same 2,330 selected steps (1,165 error + 1,165 correct/suboptimal), differing only in error-label granularity:

| File | Rows | Stages per record |
|------|-----:|---|
| `prm_0class.jsonl` | 2,330 | 1 (predicts `Correctness` only) |
| `prm_1class.jsonl` | 4,660 | 2 (`Error` then `Correctness`) |
| `prm_2class.jsonl` | 4,660 | 2 (`Math reasoning`/`Consistency` then `Correctness`) |
| `prm_4class.jsonl` | 4,660 | 2 (`Calculation`/`Method`/`Prior-step consistency`/`Domain consistency` then `Correctness`) |

Each line is a JSON object with two conversation fields plus metadata:

- `labels`: the full target sequence for training (one user turn + one assistant turn), with all label tokens filled in as ground truth.
- `inputs`: the same structure, but with the label token(s) to be predicted replaced by the placeholder `<extra>`.
- `id`: unique row id, formatted `{record_id}_{version}_{stage}`.
- `record_id`: the step's unique id after expanding PRM800K.
- `question_id`: sha1(question)[:12], shared across all rows for the same question.
- `side`: `"error"` or `"correct"`.
- `human_score`: original PRM800K rating (`1`=correct, `0`=suboptimal, `-1`=error).
- `version`: `"0class"` / `"1class"` / `"2class"` / `"4class"`.
- `stage`: `"single"` for 0-class, or `1`/`2` for the two-stage versions.

`val_question_ids.json` holds a recommended validation split: 5% of distinct `question_id`s (seed=42), shared across all four versions so results stay comparable. All rows sharing a `question_id` fall on the same side of the split.

## Acknowledgments and data lineage

This dataset is derived from:

- **PRM800K** — Lightman et al., *Let's Verify Step by Step*. [github.com/openai/prm800k](https://github.com/openai/prm800k). Released under the MIT License; this repository's dataset inherits that license (see `LICENSE`).
- **PathFinder-PRM** — Pala et al., *Error Typing for Smarter Rewards: Improving Process Reward Models with Error-Aware Hierarchical Supervision*. [github.com/declare-lab/PathFinder-PRM](https://github.com/declare-lab/PathFinder-PRM). Its error taxonomy and data-generation logic are the basis this project extends.
- **PRMBench** — Song et al., *PRMBench: A Fine-grained and Challenging Benchmark for Process-Level Reward Models*. Paper: [arxiv.org/abs/2501.03124](https://arxiv.org/abs/2501.03124); data: [huggingface.co/datasets/hitsmy/PRMBench_Preview](https://huggingface.co/datasets/hitsmy/PRMBench_Preview). Used here only to identify and exclude overlapping questions from the training sample.

## License

Released under the MIT License (see `LICENSE`). The underlying PRM800K data is itself MIT-licensed.
