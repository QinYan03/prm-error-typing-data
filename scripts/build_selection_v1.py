"""Formal record_id-only selection (selection_v1.json): no dataset formatting,
no API calls. PRMBench-overlapping questions are excluded from both sides
before selection. Error side hits ~1150 total by searching a single shared
target T for calc/method/domain top-ups (prior is always taken in full).
Correct side is drawn to match the error side's (length bucket x step bucket)
grid, with lt=0 at its natural ~13.5% share within each cell.
"""
import hashlib
import json
import random
import re
import statistics
import sys
import tiktoken
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from analyze_prm800k import load_prm800k_dataset, expand_records  # noqa: E402

# Inlined from label_pilot_4class.py / label_pilot_4class_ds.py (same tokenizer,
# same bucket thresholds) so this script no longer depends on those pilot-stage
# files.
enc = tiktoken.get_encoding("cl100k_base")
P33, P67 = 188, 327


def bucket_of(tokens):
    if tokens <= P33:
        return "short"
    if tokens <= P67:
        return "medium"
    return "long"


DATA_DIR = SCRIPT_DIR.parent
OUTPUTS_DIR = DATA_DIR / "outputs"
PRMBENCH_DIR = (DATA_DIR.parent / "PathFinder-PRM" / "src" / "evaluation" /
               "reward_guided_search" / "eval_data" / "PRMBench_Preview")
SEED = 9001
TARGET_TOTAL = 1150
TARGET_TOLERANCE = 30
LT0_SHARE = None  # computed from raw expand counts, see main()
FIELDS = ["calc_ok", "method_ok", "prior_ok", "domain_ok"]


def norm_ws(s):
    return re.sub(r"\s+", " ", s.strip())


def key_of(r):
    return (r["question"], r["prev_steps"], r["curr_step"])


def n_steps(prev_steps):
    return 0 if not prev_steps else prev_steps.count("\n\n") + 1


def step_bucket(n):
    if n <= 2:
        return "0-2"
    if n <= 5:
        return "3-5"
    return "6+"


def qid(question):
    return hashlib.sha1(question.encode("utf-8")).hexdigest()[:12]


# ---------- step 1: PRMBench audit ----------
def prmbench_audit():
    jsonl_path = PRMBENCH_DIR / "prmbench_preview.jsonl"
    json_path = PRMBENCH_DIR / "prmbench_preview.json"
    jsonl_rows = [json.loads(l) for l in open(jsonl_path, encoding="utf-8") if l.strip()]
    json_rows = json.load(open(json_path, encoding="utf-8"))

    uq_jsonl = {r["original_question"] for r in jsonl_rows if r.get("original_question")}
    uq_json = {r["question"] for r in json_rows if r.get("question")}
    union_exact = uq_jsonl | uq_json
    union_norm = {norm_ws(q) for q in union_exact}

    return {
        "jsonl_lines": len(jsonl_rows), "jsonl_unique_original_question": len(uq_jsonl),
        "json_records": len(json_rows), "json_unique_question": len(uq_json),
        "overlap_jsonl_json_exact": len(uq_jsonl & uq_json),
        "union_exact": len(union_exact), "union_normalized": len(union_norm),
    }, union_exact, union_norm


# ---------- candidate pools ----------
def load_error_pool():
    confirmed = json.loads((OUTPUTS_DIR / "labels_confirmed_v1.json").read_text(encoding="utf-8"))
    kept = [r for r in confirmed if not r["confirmed_discarded"] and r["confirmed_labels"]]
    for r in kept:
        r["tokens"] = len(enc.encode(r["question"] + r["prev_steps"] + r["curr_step"]))
        r["len_bucket"] = bucket_of(r["tokens"])
        r["steps"] = n_steps(r["prev_steps"])
        r["step_bucket"] = step_bucket(r["steps"])
    seen = {}
    for r in kept:
        k = key_of(r)
        if k not in seen or r["record_id"] < seen[k]["record_id"]:
            seen[k] = r
    return list(seen.values())


def load_correct_pool():
    dataset, _ = load_prm800k_dataset()
    records = expand_records(dataset)
    for i, r in enumerate(records):
        r["record_id"] = i
    pos = [r for r in records if r["human_score"] == 1]
    subopt = [r for r in records if r["human_score"] == 0]
    natural_lt0_share = len(subopt) / (len(pos) + len(subopt))
    for r in pos + subopt:
        r["tokens"] = len(enc.encode(r["question"] + r["prev_steps"] + r["curr_step"]))
        r["len_bucket"] = bucket_of(r["tokens"])
        r["steps"] = n_steps(r["prev_steps"])
        r["step_bucket"] = step_bucket(r["steps"])

    def dedup(recs):
        seen = {}
        for r in recs:
            k = key_of(r)
            if k not in seen or r["record_id"] < seen[k]["record_id"]:
                seen[k] = r
        return list(seen.values())

    return dedup(pos), dedup(subopt), natural_lt0_share


def exclude_prmbench(records, prmbench_norm_set):
    kept, excluded_q = [], set()
    for r in records:
        if norm_ws(r["question"]) in prmbench_norm_set:
            excluded_q.add(r["question"])
        else:
            kept.append(r)
    return kept, excluded_q


# ---------- error-side T-search ----------
def run_error_selection(pool, T):
    rng = random.Random(SEED)
    selected_ids, selected = set(), []

    def has(r, f):
        return r["confirmed_labels"][f] == 0

    for r in pool:
        if has(r, "prior_ok"):
            selected_ids.add(r["record_id"])
            selected.append(r)

    notes = []

    def top_up(field, target):
        already = sum(1 for r in selected if has(r, field))
        needed = max(0, target - already)
        candidates = [r for r in pool if has(r, field) and r["record_id"] not in selected_ids]
        rng.shuffle(candidates)
        draw = candidates[:needed]
        for r in draw:
            selected_ids.add(r["record_id"])
            selected.append(r)
        notes.append({"field": field, "target": target, "already": already, "needed": needed,
                      "available": len(candidates), "drawn": len(draw),
                      "shortfall": max(0, needed - len(draw))})

    top_up("domain_ok", T)
    top_up("calc_ok", T)
    top_up("method_ok", T)
    return selected, notes


def search_T(pool, target_total, tol):
    lo, hi = 50, 5000
    best = None
    for _ in range(40):
        mid = (lo + hi) // 2
        selected, notes = run_error_selection(pool, mid)
        total = len(selected)
        if best is None or abs(total - target_total) < abs(best[0] - target_total):
            best = (total, mid, selected, notes)
        if abs(total - target_total) <= tol:
            return total, mid, selected, notes
        if total < target_total:
            lo = mid + 1
        else:
            hi = mid - 1
        if lo > hi:
            break
    return best


# ---------- correct-side matched selection ----------
def build_correct_side(error_selected, pos_pool, subopt_pool, lt0_share):
    grid_target = Counter((r["len_bucket"], r["step_bucket"]) for r in error_selected)

    def index_by_cell(pool):
        idx = defaultdict(list)
        for r in pool:
            idx[(r["len_bucket"], r["step_bucket"])].append(r)
        return idx

    pos_idx, subopt_idx = index_by_cell(pos_pool), index_by_cell(subopt_pool)
    rng = random.Random(SEED + 1)
    selected = []
    cell_report = []

    for cell, n_cell in sorted(grid_target.items()):
        n_lt0 = round(n_cell * lt0_share)
        n_lt1 = n_cell - n_lt0
        pool_lt0 = subopt_idx.get(cell, [])
        pool_lt1 = pos_idx.get(cell, [])
        rng.shuffle(pool_lt0)
        rng.shuffle(pool_lt1)
        draw_lt0 = pool_lt0[:n_lt0]
        draw_lt1 = pool_lt1[:n_lt1]
        selected.extend(draw_lt0)
        selected.extend(draw_lt1)
        cell_report.append({
            "cell": f"{cell[0]}/{cell[1]}", "target": n_cell,
            "lt0_target": n_lt0, "lt0_available": len(pool_lt0), "lt0_drawn": len(draw_lt0),
            "lt1_target": n_lt1, "lt1_available": len(pool_lt1), "lt1_drawn": len(draw_lt1),
            "shortfall": max(0, n_lt0 - len(draw_lt0)) + max(0, n_lt1 - len(draw_lt1)),
        })
    return selected, cell_report


def dist_stats(values):
    if not values:
        return None
    values = sorted(values)
    return {"median": statistics.median(values), "p25": values[int(len(values) * 0.25)],
            "p75": values[int(len(values) * 0.75)], "n": len(values)}


def recheck_reproduction_map():
    b1 = json.loads((OUTPUTS_DIR / "batch1_results.json").read_text(encoding="utf-8"))
    b2 = json.loads((OUTPUTS_DIR / "batch2_results.json").read_text(encoding="utf-8"))
    b3 = json.loads((OUTPUTS_DIR / "batch3_results.json").read_text(encoding="utf-8"))
    by_id_12 = {r["record_id"]: r for r in b1 + b2}
    by_id_3 = {r["record_id"]: r for r in b3}

    a1 = json.loads((OUTPUTS_DIR / "recheck_prior_results_a.json").read_text(encoding="utf-8"))
    b1r = json.loads((OUTPUTS_DIR / "recheck_prior_results_b.json").read_text(encoding="utf-8"))
    a3 = json.loads((OUTPUTS_DIR / "recheck_prior_batch3_results_a.json").read_text(encoding="utf-8"))
    b3r = json.loads((OUTPUTS_DIR / "recheck_prior_batch3_results_b.json").read_text(encoding="utf-8"))

    repro = {}

    def ingest(records_a, records_b, by_id):
        b_by_id = {r["record_id"]: r for r in records_b}
        for ra in records_a:
            rid = ra["record_id"]
            rb = b_by_id.get(rid)
            a0 = bool(ra["labels"] and ra["labels"]["prior_ok"] == 0)
            b0 = bool(rb and rb["labels"] and rb["labels"]["prior_ok"] == 0)
            repro[key_of(by_id[rid])] = int(a0) + int(b0)

    ingest(a1, b1r, by_id_12)
    ingest(a3, b3r, by_id_3)
    return repro


def main():
    print("Step 1: PRMBench audit...", flush=True)
    audit, prmbench_exact, prmbench_norm = prmbench_audit()

    print("Step 2: loading pools...", flush=True)
    error_pool_all = load_error_pool()
    pos_pool_all, subopt_pool_all, natural_lt0_share = load_correct_pool()

    n_prior_before = sum(1 for r in error_pool_all if r["confirmed_labels"]["prior_ok"] == 0)
    error_pool, excluded_error_q = exclude_prmbench(error_pool_all, prmbench_norm)
    pos_pool, excluded_pos_q = exclude_prmbench(pos_pool_all, prmbench_norm)
    subopt_pool, excluded_subopt_q = exclude_prmbench(subopt_pool_all, prmbench_norm)
    n_prior_after = sum(1 for r in error_pool if r["confirmed_labels"]["prior_ok"] == 0)

    exact_excluded_error = sum(1 for r in error_pool_all if r["question"] in prmbench_exact)
    norm_excluded_error = len(error_pool_all) - len(error_pool)

    print("Step 3: searching T for error-side total ~%d..." % TARGET_TOTAL, flush=True)
    total, T, error_selected, notes = search_T(error_pool, TARGET_TOTAL, TARGET_TOLERANCE)
    print(f"  T={T}, total={total}", flush=True)

    print("Step 4: correct-side matched selection...", flush=True)
    correct_selected, cell_report = build_correct_side(error_selected, pos_pool, subopt_pool, natural_lt0_share)

    print("Step 5: recheck reproduction map...", flush=True)
    repro_map = recheck_reproduction_map()

    # ---- selection_v1.json ----
    out_records = []
    for r in error_selected:
        out_records.append({
            "record_id": r["record_id"], "side": "error", "human_score": r["human_score"],
            "question_id": qid(r["question"]), "len_bucket": r["len_bucket"],
            "step_bucket": r["step_bucket"], "confirmed_labels": r["confirmed_labels"],
            "prior_reproduction_count": (repro_map.get(key_of(r)) if r["confirmed_labels"]["prior_ok"] == 0 else None),
        })
    for r in correct_selected:
        out_records.append({
            "record_id": r["record_id"], "side": "correct", "human_score": r["human_score"],
            "question_id": qid(r["question"]), "len_bucket": r["len_bucket"],
            "step_bucket": r["step_bucket"],
        })
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "selection_v1.json").write_text(
        json.dumps({"seed": SEED, "error_T": T, "error_total": len(error_selected),
                    "correct_total": len(correct_selected), "records": out_records},
                   indent=2, ensure_ascii=False),
        encoding="utf-8")

    # ---- stats for report ----
    err_field_counts = {f: sum(1 for r in error_selected if r["confirmed_labels"][f] == 0) for f in FIELDS}
    err_grid = Counter((r["len_bucket"], r["step_bucket"]) for r in error_selected)
    err_steps = [r["steps"] for r in error_selected]
    err_tokens = [r["tokens"] for r in error_selected]
    cor_steps = [r["steps"] for r in correct_selected]
    cor_tokens = [r["tokens"] for r in correct_selected]
    n_lt0_selected = sum(1 for r in correct_selected if r["human_score"] == 0)

    all_selected = error_selected + correct_selected
    q_all = Counter(r["question"] for r in all_selected)
    n_distinct_q = len(q_all)
    n_repeated_q = sum(1 for c in q_all.values() if c >= 2)
    n_records_repeated_q = sum(c for c in q_all.values() if c >= 2)
    err_q = set(r["question"] for r in error_selected)
    cor_q = set(r["question"] for r in correct_selected)
    n_both_sides = len(err_q & cor_q)

    n_overlap_final = sum(1 for r in all_selected if norm_ws(r["question"]) in prmbench_norm)

    report = {
        "audit": audit, "T": T, "error_total": len(error_selected), "correct_total": len(correct_selected),
        "n_prior_before_exclusion": n_prior_before, "n_prior_after_exclusion": n_prior_after,
        "exact_vs_norm_excluded_error_side": {"exact": exact_excluded_error, "normalized": norm_excluded_error},
        "excluded_question_counts": {"error_pool": len(excluded_error_q), "pos_pool": len(excluded_pos_q),
                                     "subopt_pool": len(excluded_subopt_q)},
        "err_field_counts": err_field_counts, "err_grid": {f"{k[0]}/{k[1]}": v for k, v in err_grid.items()},
        "notes": notes, "cell_report": cell_report, "n_lt0_selected": n_lt0_selected,
        "natural_lt0_share": natural_lt0_share,
        "err_steps_dist": dist_stats(err_steps), "err_tokens_dist": dist_stats(err_tokens),
        "cor_steps_dist": dist_stats(cor_steps), "cor_tokens_dist": dist_stats(cor_tokens),
        "n_distinct_q": n_distinct_q, "n_repeated_q": n_repeated_q, "n_records_repeated_q": n_records_repeated_q,
        "n_both_sides": n_both_sides, "n_overlap_final_with_prmbench": n_overlap_final,
    }
    (OUTPUTS_DIR / "selection_v1_report_data.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Wrote selection_v1.json and selection_v1_report_data.json")
    print(json.dumps({k: v for k, v in report.items() if k not in ("notes", "cell_report")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
