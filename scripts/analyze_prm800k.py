"""
Task 1: replicate PathFinder-PRM's src/data_generation/load_data.py expansion logic
over the 4 PRM800K jsonl files, then compute:
  - counts/proportions of human_score (rating) in {1, 0, -1, other}
  - for rating == -1 records: token-length distribution of question+prev_steps+curr_step
"""
import json
import os
from pathlib import Path
import tiktoken

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent
OUTPUTS_DIR = DATA_DIR / "outputs"

DATASET_DIR = Path(os.environ.get("PRM800K_DIR", str(DATA_DIR / "PRM800K")))
FILES = ["phase1_train.jsonl", "phase1_test.jsonl", "phase2_train.jsonl", "phase2_test.jsonl"]

enc = tiktoken.get_encoding("cl100k_base")  # GPT-3.5/4 tokenizer, used only as a consistent proxy

def load_prm800k_dataset():
    dataset = []
    per_file_counts = {}
    for file_name in FILES:
        file_path = DATASET_DIR / file_name
        n = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                dataset.append(json.loads(line))
                n += 1
        per_file_counts[file_name] = n
    return dataset, per_file_counts


def expand_records(dataset):
    """Mirrors create_new_dataset() in load_data.py exactly (rating counting + text fields only,
    we skip writing score_A/B/C/context since we only need the stats)."""
    records = []
    for example in dataset:
        question_text = example["question"]["problem"]
        steps = example["label"]["steps"]
        previous_steps = []

        for step in steps:
            completions = step.get("completions") or []

            for completion in completions:
                context = ""
                if previous_steps:
                    context += "\n\n".join(previous_steps)

                step_text = completion.get("text", "")

                records.append({
                    "question": question_text,
                    "prev_steps": context,
                    "curr_step": step_text,
                    "human_score": completion.get("rating"),
                })

            chosen_idx = step.get("chosen_completion")
            chosen_text = ""

            if chosen_idx is not None and 0 <= chosen_idx < len(completions):
                chosen_text = completions[chosen_idx].get("text", "")
            elif step.get("human_completion"):
                chosen_text = step["human_completion"].get("text", "")
            else:
                continue

            previous_steps.append(chosen_text)

    return records


def main():
    print("Loading PRM800K jsonl files...")
    dataset, per_file_counts = load_prm800k_dataset()
    print("Per-file raw question counts:", per_file_counts)
    print(f"Total raw question/completion-tree examples loaded: {len(dataset)}")

    print("\nExpanding into per-completion records (load_data.py logic)...")
    records = expand_records(dataset)
    print(f"Total expanded records: {len(records)}")

    # --- Task 1a: rating distribution ---
    from collections import Counter
    rating_counts = Counter(r["human_score"] for r in records)
    total = len(records)

    print("\n=== human_score (rating) distribution ===")
    for key in sorted(rating_counts.keys(), key=lambda x: (x is None, x)):
        cnt = rating_counts[key]
        print(f"  rating={key!r}: {cnt} ({100*cnt/total:.2f}%)")

    # --- Task 1b: token length distribution for rating == -1 ---
    neg1_records = [r for r in records if r["human_score"] == -1]
    print(f"\n=== Token length stats for rating=-1 records (n={len(neg1_records)}) ===")

    lengths = []
    for r in neg1_records:
        combined = r["question"] + "\n\n" + r["prev_steps"] + "\n\n" + r["curr_step"]
        n_tokens = len(enc.encode(combined))
        lengths.append(n_tokens)

    lengths.sort()
    import statistics

    def pct(p):
        if not lengths:
            return None
        idx = min(len(lengths) - 1, int(round(p * (len(lengths) - 1))))
        return lengths[idx]

    stats = {
        "n": len(lengths),
        "min": lengths[0] if lengths else None,
        "P50": pct(0.50),
        "P75": pct(0.75),
        "P90": pct(0.90),
        "P95": pct(0.95),
        "P99": pct(0.99),
        "max": lengths[-1] if lengths else None,
        "mean": round(statistics.mean(lengths), 1) if lengths else None,
    }
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # bucket distribution for report readability
    bucket_edges = [256, 512, 1024, 2048, 4096, 8192]
    bucket_labels = ["<=256", "257-512", "513-1024", "1025-2048", "2049-4096", "4097-8192", ">8192"]
    bucket_counts = [0] * len(bucket_labels)
    import bisect
    for n in lengths:
        i = bisect.bisect_right(bucket_edges, n)
        bucket_counts[i] += 1
    print("\n  Token length buckets (rating=-1):")
    for label, cnt in zip(bucket_labels, bucket_counts):
        print(f"    {label:>10}: {cnt:>6} ({100*cnt/len(lengths):.2f}%)")

    # Save raw numbers to json for report-writing
    out = {
        "per_file_raw_counts": per_file_counts,
        "total_raw_examples": len(dataset),
        "total_expanded_records": total,
        "rating_counts": {str(k): v for k, v in rating_counts.items()},
        "rating_pct": {str(k): round(100*v/total, 3) for k, v in rating_counts.items()},
        "neg1_token_stats": stats,
        "neg1_token_buckets": dict(zip(bucket_labels, bucket_counts)),
    }
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "prm800k_stats.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved summary stats to {out_path}")

    # also dump a handful of real rating=-1 samples for task 2 (LLM testing)
    import random
    random.seed(42)
    sample_pool = [r for r in neg1_records if len(enc.encode(r["question"]+r["prev_steps"]+r["curr_step"])) < 1500]
    samples = random.sample(sample_pool, min(5, len(sample_pool)))
    samples_path = OUTPUTS_DIR / "neg1_samples.json"
    with open(samples_path, "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(samples)} sample rating=-1 records to {samples_path}")


if __name__ == "__main__":
    main()
