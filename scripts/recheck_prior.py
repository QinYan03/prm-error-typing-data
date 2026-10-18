"""Step 4: prior_ok=0 recheck test.

Object: among the cumulative batch1+batch2 (4500) kept records with prior_ok=0,
dedup by (question, prev_steps, curr_step) taking one representative per group
(same rule as build_dedup_stats.py: smallest order_index).

Two independent re-labeling passes on the same v3 prompt, single parseable draw
per record (parse-failure retries only, no disagreement-retry loop -- this is
measuring reproducibility, not re-filtering):
  (a) thinking off, temperature=1.0 (same calling convention as batch1/batch2)
  (b) thinking on, API default effort, no temperature sent (same convention as
      the pilot stage's "think" runs)

call_deepseek's hard-stop-on-reasoning-with-thinking-off guard is controlled by
cfg.thinking (an explicit parameter); mode (b) uses cfg.thinking="on", which by
construction does not trigger that guard. Mode (a) keeps the guard active
(default behavior unchanged).

Prints an estimated cost (from observed batch1 nothink / v3 think token
averages) before making any call; if the estimate exceeds $2, writes a
"paused" report and makes no API calls at all.
"""
import json
import sys
import threading
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from label_pilot_4class import make_full_context  # noqa: E402
from label_pilot_4class_ds import (  # noqa: E402
    BASE_URL, REQUEST_TIMEOUT, RunConfig, RunState, FatalAPIError, TransientExhausted,
    get_labels_with_parse_retry, fetch_balance,
)
from prompt_4class_v3 import CLS_PROMPT_4CLASS  # noqa: E402

DATA_DIR = SCRIPT_DIR.parent
OUTPUTS_DIR = DATA_DIR / "outputs"
COST_LIMIT_USD = 2.0
WORKERS = 4
PRICE_IN, PRICE_OUT = 0.15, 0.6
# observed averages from already-run stages, used only for the pre-call estimate
EST_PROMPT_A, EST_COMPLETION_A = 1066, 793     # batch1, thinking off
EST_PROMPT_B, EST_COMPLETION_B = 1107, 4398    # v3 pilot "think" run, thinking on


def key_of(r):
    return (r["question"], r["prev_steps"], r["curr_step"])


def load_cumulative():
    batch1 = json.loads((OUTPUTS_DIR / "batch1_results.json").read_text(encoding="utf-8"))
    batch2 = json.loads((OUTPUTS_DIR / "batch2_results.json").read_text(encoding="utf-8"))
    return batch1 + batch2


def dedup_reps(records):
    groups = defaultdict(list)
    for r in records:
        groups[key_of(r)].append(r)
    return [sorted(v, key=lambda r: r["order_index"])[0] for v in groups.values()]


def relabel_once(client, record, state):
    full_context = make_full_context(record["question"], record["prev_steps"], record["curr_step"])
    prompt = CLS_PROMPT_4CLASS.format(context=full_context)
    calls = []
    labels, content = get_labels_with_parse_retry(client, prompt, state, calls, "recheck")
    return {"record_id": record["record_id"], "labels": labels, "raw_reasoning": content,
            "model": calls[-1]["model"] if calls else None, "calls": calls}


def run_mode(client, cfg, reps, mode_label):
    state = RunState(cfg, 0, 0)
    out = [None] * len(reps)
    todo_iter = iter(enumerate(reps))
    lock = threading.Lock()

    def worker():
        while True:
            with lock:
                item = next(todo_iter, None)
            if item is None:
                return
            i, record = item
            print(f"[{mode_label} {i+1}/{len(reps)}] record_id={record['record_id']}", flush=True)
            try:
                out[i] = relabel_once(client, record, state)
            except (FatalAPIError, TransientExhausted) as e:
                print(f"[{mode_label} FAILED] record_id={record['record_id']}: {e}", flush=True)
                out[i] = {"record_id": record["record_id"], "labels": None, "raw_reasoning": None,
                          "model": None, "calls": [], "error": str(e)}

    threads = [threading.Thread(target=worker) for _ in range(WORKERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


def pct(x):
    return "n/a" if x is None else f"{x * 100:.1f}%"


def build_report(reps, results_a, results_b, prior30_ids):
    a_by_id = {r["record_id"]: r for r in results_a}
    b_by_id = {r["record_id"]: r for r in results_b}

    def matches_prior0(labels):
        return labels is not None and labels.get("prior_ok") == 0

    def matches_exact(orig, labels):
        return labels is not None and all(orig[f] == labels[f] for f in ["calc_ok", "method_ok", "prior_ok", "domain_ok"])

    n = len(reps)
    a_prior0 = sum(1 for r in reps if matches_prior0(a_by_id[r["record_id"]]["labels"]))
    b_prior0 = sum(1 for r in reps if matches_prior0(b_by_id[r["record_id"]]["labels"]))
    a_exact = sum(1 for r in reps if matches_exact(r["final_labels"], a_by_id[r["record_id"]]["labels"]))
    b_exact = sum(1 for r in reps if matches_exact(r["final_labels"], b_by_id[r["record_id"]]["labels"]))

    L = ["# prior_ok=0 recheck report", "",
         f"Scope: kept records with prior_ok=0 among the cumulative 4500 (batch1+batch2), "
         f"deduplicated by (question, prev_steps, curr_step) (smallest order_index per group), "
         f"{n} records in total. Two recheck modes: (a) thinking off, temperature=1.0, independent "
         f"resample, (b) thinking on (API default). Both are single parseable draws (parse-failure "
         f"retries only, no disagreement-retry loop).", ""]

    L.append("## Reproduction rate")
    L.append("")
    L.append("| Mode | prior_ok=0 reproduction rate | All-four-labels exact match rate |")
    L.append("|---|---:|---:|")
    L.append(f"| (a) thinking off | {a_prior0}/{n} = {pct(a_prior0/n if n else None)} | {a_exact}/{n} = {pct(a_exact/n if n else None)} |")
    L.append(f"| (b) thinking on | {b_prior0}/{n} = {pct(b_prior0/n if n else None)} | {b_exact}/{n} = {pct(b_exact/n if n else None)} |")
    L.append("")

    L.append("## Per-record comparison")
    L.append("")
    L.append("| record_id | in prior30 sample | original labels | (a) labels | (b) labels |")
    L.append("|---|---|---|---|---|")
    for r in reps:
        mark = "yes" if r["record_id"] in prior30_ids else ""
        orig = r["final_labels"]
        la = a_by_id[r["record_id"]]["labels"]
        lb = b_by_id[r["record_id"]]["labels"]

        def fmt(lab):
            if lab is None:
                return "parse failed"
            return f"C{lab['calc_ok']}M{lab['method_ok']}P{lab['prior_ok']}D{lab['domain_ok']}"

        L.append(f"| {r['record_id']} | {mark} | {fmt(orig)} | {fmt(la)} | {fmt(lb)} |")
    L.append("")
    return "\n".join(L)


def main():
    cumulative = load_cumulative()
    kept_prior0 = [r for r in cumulative if not r["discarded"] and r["final_labels"]
                   and r["final_labels"]["prior_ok"] == 0]
    reps = dedup_reps(kept_prior0)
    n = len(reps)

    cost_a = n * (EST_PROMPT_A * PRICE_IN + EST_COMPLETION_A * PRICE_OUT) / 1e6
    cost_b = n * (EST_PROMPT_B * PRICE_IN + EST_COMPLETION_B * PRICE_OUT) / 1e6
    est_total = cost_a + cost_b
    print(f"n_groups={n}; estimated cost: mode(a) ${cost_a:.3f}, mode(b) ${cost_b:.3f}, "
          f"total ${est_total:.3f} (limit ${COST_LIMIT_USD})", flush=True)

    estimate = {"n_groups": n, "cost_a_usd": round(cost_a, 4), "cost_b_usd": round(cost_b, 4),
                "total_usd": round(est_total, 4), "limit_usd": COST_LIMIT_USD,
                "assumptions": {"est_prompt_a": EST_PROMPT_A, "est_completion_a": EST_COMPLETION_A,
                                "est_prompt_b": EST_PROMPT_B, "est_completion_b": EST_COMPLETION_B,
                                "price_in_per_1m": PRICE_IN, "price_out_per_1m": PRICE_OUT}}
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "recheck_prior_cost_estimate.json").write_text(
        json.dumps(estimate, indent=2, ensure_ascii=False), encoding="utf-8")

    if est_total > COST_LIMIT_USD:
        (OUTPUTS_DIR / "recheck_prior_report.md").write_text(
            f"# prior_ok=0 recheck test - paused, no API calls made\n\n"
            f"Estimated cost ${est_total:.3f} exceeds the configured limit ${COST_LIMIT_USD}; "
            f"no API calls were made.\n\n"
            f"Records pending recheck after deduplication: {n}\n\n"
            f"See `recheck_prior_cost_estimate.json` for the cost breakdown.\n",
            encoding="utf-8")
        print("PAUSED: estimated cost exceeds limit, no API calls made.", flush=True)
        return

    load_dotenv(DATA_DIR / ".env")
    import os
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=BASE_URL, max_retries=0, timeout=REQUEST_TIMEOUT)

    cfg_a = RunConfig("off", "_recheck_a", {"thinking": {"type": "disabled"}}, 1.0, False, None)
    cfg_b = RunConfig("on", "_recheck_b", None, None, False, None)

    results_a = run_mode(client, cfg_a, reps, "a")
    (OUTPUTS_DIR / "recheck_prior_results_a.json").write_text(
        json.dumps(results_a, indent=2, ensure_ascii=False), encoding="utf-8")

    results_b = run_mode(client, cfg_b, reps, "b")
    (OUTPUTS_DIR / "recheck_prior_results_b.json").write_text(
        json.dumps(results_b, indent=2, ensure_ascii=False), encoding="utf-8")

    import re
    prior30_path = OUTPUTS_DIR / "review_full_prior30.md"
    if prior30_path.exists():
        prior30_text = prior30_path.read_text(encoding="utf-8")
        prior30_ids = {int(x) for x in re.findall(r"record_id=(\d+)", prior30_text)}
    else:
        print(f"[skip] {prior30_path.name} not found, the 'in prior30 sample' column will be blank for all rows.", flush=True)
        prior30_ids = set()

    report = build_report(reps, results_a, results_b, prior30_ids)
    (OUTPUTS_DIR / "recheck_prior_report.md").write_text(report, encoding="utf-8")

    actual_calls = [c for r in results_a + results_b for c in r["calls"]]
    actual_cost = sum((c["prompt_tokens"] * PRICE_IN + c["completion_tokens"] * PRICE_OUT) / 1e6 for c in actual_calls)
    print(f"Done. n_groups={n}, actual cost=${actual_cost:.4f}", flush=True)


if __name__ == "__main__":
    main()
