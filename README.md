# Fine-Grained Error Typing Data for Process Reward Models

A data pipeline and dataset for training **process reward models (PRMs)** with a **four-category error taxonomy**, built from the PRM800K step-level math reasoning dataset.

This repository contains my part, **data construction**, of a team course project (*Process Reward Modeling in LLMs*, Heidelberg University, 2026). Model training and evaluation were done by teammates and are not included here.

**Full method write-up:** [`report/data_construction.pdf`](report/data_construction.pdf)

## At a glance

- **Goal:** extend PathFinder-PRM's two error categories (Math, Consistency) into four (Calculation, Method, Prior-step consistency, Domain consistency), and build training data at four label granularities (0/1/2/4-class) for a controlled comparison.
- **Scale:** 7,500 human-rated error steps labeled with an LLM (DeepSeek API). 5,952 unique steps kept after consistency filtering. Final balanced dataset of 2,330 steps in four training versions.
- **Quality control:** LLM labels filtered against human ratings. For the least reliable category (Prior-step consistency), a reproducibility check raised the acceptable-label rate in spot checks from 50% to 83%.
- **Leakage control:** all questions overlapping with the PRMBench evaluation benchmark removed before sampling.
- **Tools:** Python, LLM API annotation, prompt design, JSONL data pipelines.

## Pipeline

```
PRM800K (human step ratings)
  -> expand solutions into per-step records
  -> LLM labeling of error steps into 4 categories, discard labels that contradict the human rating
  -> re-label Prior-step consistency errors twice, keep only labels reproduced at least once
  -> exclude PRMBench questions, sample error : correct = 1 : 1, matched on context length and step count
  -> format into 0/1/2/4-class training files with a shared validation split
```

## Key design decisions

- **Grounded taxonomy.** The four categories split criteria already present in the original PathFinder-PRM annotation guidelines (9 of 10 criteria are taken from them) instead of inventing new error types.
- **Human labels first.** Only steps that humans rated as errors are sent to the LLM. Correct and suboptimal steps get their labels directly from the human ratings, following the original pipeline.
- **One labeling pass, four versions.** The 1-class and 2-class labels are derived from the same 4-class labels by logical OR, so the four versions differ only in label granularity, not in labeling noise.
- **Balanced, shortcut-resistant sampling.** Errors are oversampled to 1:1 so that rare categories have enough examples. Correct steps are matched to error steps on context length and number of prior steps, because PRM800K annotators stop at the first error and correct steps would otherwise be systematically shorter.

## Dataset (`dataset_v1/`)

All four files are built from the same 2,330 steps (1,165 error, 1,165 correct or suboptimal). Error side by category: Calculation 503, Method 493, Prior-step consistency 75, Domain consistency 498 (a step with several categories is counted in each).

| File | Rows | Prediction target |
|------|-----:|---|
| `prm_0class.jsonl` | 2,330 | Correctness only (single stage) |
| `prm_1class.jsonl` | 4,660 | Error, then Correctness |
| `prm_2class.jsonl` | 4,660 | Math / Consistency, then Correctness |
| `prm_4class.jsonl` | 4,660 | Calculation / Method / Prior-step / Domain, then Correctness |

Each row has a `labels` field (target sequence with `<+>` / `<->` label tokens) and an `inputs` field (same sequence with labels replaced by `<extra>`), plus metadata (`id`, `record_id`, `question_id`, `side`, `human_score`, `version`, `stage`). `val_question_ids.json` defines a recommended validation split (5% of questions, seed 42) shared by all four versions.

## Reproducing the pipeline

**Setup**

```bash
pip install -r requirements.txt
cp .env.example .env    # then set DEEPSEEK_API_KEY in .env
```

Download the four PRM800K files (`phase1_train/test.jsonl`, `phase2_train/test.jsonl`) from [openai/prm800k](https://github.com/openai/prm800k) into `PRM800K/`, or set `PRM800K_DIR`. The PRMBench exclusion step expects a checkout of [PathFinder-PRM](https://github.com/declare-lab/PathFinder-PRM) next to this repository.

**Run** (from the repository root). Steps marked *API* call DeepSeek and incur cost.

```bash
python scripts/analyze_prm800k.py                                                  # load and expand PRM800K
python scripts/label_batch_4class_ds.py --start 0    --end 2500 --batch-name batch1   # API: LLM labeling
python scripts/label_batch_4class_ds.py --start 2500 --end 4500 --batch-name batch2   # API
python scripts/label_batch_4class_ds.py --start 4500 --end 7500 --batch-name batch3   # API
python scripts/recheck_prior.py                                                    # API: Prior-step re-labeling, batches 1-2
python scripts/recheck_prior_batch3.py                                             # API: Prior-step re-labeling, batch 3
python scripts/build_confirmed_labels.py                                           # apply re-labeling results
python scripts/build_selection_v1.py                                               # PRMBench exclusion and balanced sampling
python scripts/build_dataset_v1.py                                                 # write the four training files
```

Notes: intermediate files go to `outputs/`. `outputs/pilot_4class_sample_ds.json` is a frozen list of pilot samples excluded from labeling. Labeling runs resume from records already written, and the re-labeling scripts stop if their cost estimate exceeds $2. The labeling prompt is defined in `scripts/prompt_4class_v3.py`. The remaining files in `scripts/` are helper modules imported by the pipeline.

## Acknowledgments

- **PRM800K**: Lightman et al., *Let's Verify Step by Step* ([repo](https://github.com/openai/prm800k), MIT License). Source of all steps and human ratings.
- **PathFinder-PRM**: Pala et al., *Error Typing for Smarter Rewards* ([repo](https://github.com/declare-lab/PathFinder-PRM)). The error taxonomy and data-generation logic this project extends.
- **PRMBench**: Song et al., *PRMBench* ([paper](https://arxiv.org/abs/2501.03124), [data](https://huggingface.co/datasets/hitsmy/PRMBench_Preview)). Used only to exclude overlapping questions.

## License

MIT (see `LICENSE`). The underlying PRM800K data is also MIT-licensed.
