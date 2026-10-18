"""Full generation of the four training-format JSONL files (0/1/2/4 class) plus
the recommended validation-question-id split. No API calls.

Row construction (fields, token conventions, system prompts, label aggregation)
is byte-for-byte the same logic as build_format_preview.py;
this script just applies it to the full selection_v1.json instead of 4 examples.
"""
import json
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from analyze_prm800k import load_prm800k_dataset, expand_records  # noqa: E402
from build_format_preview import build_all_versions  # noqa: E402

DATA_DIR = SCRIPT_DIR.parent
OUTPUTS_DIR = DATA_DIR / "outputs"
DATASET_DIR = DATA_DIR / "dataset_v1"
VAL_FRACTION = 0.05
VAL_SEED = 42
VERSIONS = ["0class", "1class", "2class", "4class"]
FILENAMES = {"0class": "prm_0class.jsonl", "1class": "prm_1class.jsonl",
            "2class": "prm_2class.jsonl", "4class": "prm_4class.jsonl"}


def load_records():
    sel = json.loads((OUTPUTS_DIR / "selection_v1.json").read_text(encoding="utf-8"))["records"]
    confirmed = {r["record_id"]: r for r in json.loads((OUTPUTS_DIR / "labels_confirmed_v1.json")
                                                        .read_text(encoding="utf-8"))}
    dataset, _ = load_prm800k_dataset()
    expanded = expand_records(dataset)

    full_records = []
    for ex in sel:
        rid = ex["record_id"]
        if ex["side"] == "error":
            full = confirmed[rid]
            rec = {"question": full["question"], "prev_steps": full["prev_steps"],
                  "curr_step": full["curr_step"], "human_score": full["human_score"],
                  "labels": full["confirmed_labels"]}
        else:
            full = expanded[rid]
            rec = {"question": full["question"], "prev_steps": full["prev_steps"],
                  "curr_step": full["curr_step"], "human_score": full["human_score"], "labels": None}
        meta = {"record_id": rid, "question_id": ex["question_id"], "side": ex["side"]}
        full_records.append((rec, meta))

    full_records.sort(key=lambda x: x[1]["record_id"])
    return full_records


def main():
    DATASET_DIR.mkdir(exist_ok=True)
    full_records = load_records()
    print(f"Loaded {len(full_records)} selected records (sorted by record_id).")

    writers = {v: open(DATASET_DIR / FILENAMES[v], "w", encoding="utf-8") for v in VERSIONS}
    counts = {v: 0 for v in VERSIONS}

    for rec, meta in full_records:
        rows, _ = build_all_versions(rec, meta)
        for v in VERSIONS:
            for row in rows[v]:
                writers[v].write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[v] += 1

    for f in writers.values():
        f.close()
    print("Row counts:", counts)

    # ---- validation split by question_id, shared across all versions ----
    distinct_qids = sorted({meta["question_id"] for _, meta in full_records})
    n_val = round(len(distinct_qids) * VAL_FRACTION)
    rng = random.Random(VAL_SEED)
    val_qids = set(rng.sample(distinct_qids, n_val))
    val_payload = {
        "seed": VAL_SEED, "fraction": VAL_FRACTION,
        "n_total_questions": len(distinct_qids), "n_val_questions": len(val_qids),
        "val_question_ids": sorted(val_qids),
    }
    (DATASET_DIR / "val_question_ids.json").write_text(
        json.dumps(val_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Validation split: {len(val_qids)}/{len(distinct_qids)} questions "
          f"({len(val_qids)/len(distinct_qids)*100:.1f}%)")


if __name__ == "__main__":
    main()
