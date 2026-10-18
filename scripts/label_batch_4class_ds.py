"""Formal batch labeling with prompt_4class_v3.py (DeepSeek, thinking off).

Candidate pool: all PRM800K human_score==-1 records (official load_data.py
expansion logic), excluding the 200 record_ids already used in the pilot
(pilot_4class_sample_ds.json). Fixed-seed shuffle, saved once to
batch_candidates.json; --start/--end selects a slice of that ordered list for
a given batch run (batch1 = [0:2500]).

Reuses call/backoff/retry machinery from label_pilot_4class_ds.py verbatim
(RunConfig, RunState, call_deepseek, get_labels_with_parse_retry, ...).

Guardrails (halt, keep partial results, distinct from the per-record retry logic):
  - after the 100th completed record: discard rate outside [8%, 35%]
  - cumulative parse-failure rate > 2% (checked once >=20 parse attempts logged)
  - average completion tokens per call > 2000 (RunState's own avg_check/avg_limit;
    also call_deepseek independently hard-stops the instant any single call shows
    nonzero reasoning tokens despite thinking being disabled)
"""
import argparse
import hashlib
import json
import os
import random
import sys
import threading
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from analyze_prm800k import load_prm800k_dataset, expand_records  # noqa: E402
from label_pilot_4class import enc, make_full_context, MAX_DISAGREEMENT_RETRIES  # noqa: E402
from label_pilot_4class_ds import (  # noqa: E402
    BASE_URL, REQUEST_TIMEOUT, P33, P67, bucket_of,
    RunConfig, RunState, FatalAPIError, TransientExhausted,
    get_labels_with_parse_retry, load_done_results, fetch_balance, utcnow,
)
from prompt_4class_v3 import CLS_PROMPT_4CLASS  # noqa: E402

DATA_DIR = SCRIPT_DIR.parent
OUTPUTS_DIR = DATA_DIR / "outputs"
PROMPT_FILE = SCRIPT_DIR / "prompt_4class_v3.py"
EXPECTED_PROMPT_SHA8 = "d62235bd"
SEED_BATCH = 2025
WORKERS = 8

CANDIDATES_PATH = OUTPUTS_DIR / "batch_candidates.json"
PILOT_SAMPLE_PATH = OUTPUTS_DIR / "pilot_4class_sample_ds.json"

DISCARD_RATE_CHECK_N = 100
DISCARD_RATE_MIN, DISCARD_RATE_MAX = 0.08, 0.35
PARSE_FAIL_RATE_MAX = 0.02
PARSE_FAIL_MIN_ATTEMPTS = 20
AVG_TOKEN_LIMIT = 2000


def build_candidates():
    dataset, _ = load_prm800k_dataset()
    records = expand_records(dataset)
    neg1 = [dict(r, record_id=i) for i, r in enumerate(records) if r["human_score"] == -1]
    for r in neg1:
        r["tokens"] = len(enc.encode(r["question"] + r["prev_steps"] + r["curr_step"]))
        r["bucket"] = bucket_of(r["tokens"])

    pilot_ids = {r["record_id"] for r in json.loads(PILOT_SAMPLE_PATH.read_text(encoding="utf-8"))}
    pool = [r for r in neg1 if r["record_id"] not in pilot_ids]

    key_counts = Counter((r["question"], r["prev_steps"], r["curr_step"]) for r in pool)
    dup_groups = {k: c for k, c in key_counts.items() if c > 1}
    n_dup_records = sum(dup_groups.values())

    rng = random.Random(SEED_BATCH)
    order = list(range(len(pool)))
    rng.shuffle(order)
    ordered = [pool[i] for i in order]
    for idx, r in enumerate(ordered):
        r["order_index"] = idx

    meta = {
        "seed": SEED_BATCH,
        "total_neg1_records": len(neg1),
        "pilot_excluded_count": len(pilot_ids),
        "pool_size": len(pool),
        "duplicate_tuple_groups": len(dup_groups),
        "duplicate_tuple_records": n_dup_records,
    }
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATES_PATH.write_text(json.dumps({"meta": meta, "candidates": ordered}, ensure_ascii=False), encoding="utf-8")
    return meta, ordered


def load_candidates():
    if CANDIDATES_PATH.exists():
        payload = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))
        return payload["meta"], payload["candidates"]
    return build_candidates()


def label_record(client, record, state):
    full_context = make_full_context(record["question"], record["prev_steps"], record["curr_step"])
    prompt = CLS_PROMPT_4CLASS.format(context=full_context)
    calls = []

    labels, content = get_labels_with_parse_retry(client, prompt, state, calls, "initial")

    result = {
        "record_id": record["record_id"],
        "order_index": record["order_index"],
        "bucket": record["bucket"],
        "tokens": record["tokens"],
        "question": record["question"],
        "prev_steps": record["prev_steps"],
        "curr_step": record["curr_step"],
        "human_score": record["human_score"],
        "thinking": state.cfg.thinking,
        "final_labels": labels,
        "correctness": None,
        "raw_reasoning": content,
        "model": calls[-1]["model"] if calls else None,
        "n_disagreement_retries": 0,
        "discarded": False,
        "discard_reason": None,
        "calls": calls,
    }

    if labels is None:
        result["discarded"] = True
        result["discard_reason"] = "unparseable"
        return result

    retries = 0
    while all(v == 1 for v in labels.values()) and retries < MAX_DISAGREEMENT_RETRIES:
        retries += 1
        retry_labels, retry_content = get_labels_with_parse_retry(
            client, prompt, state, calls, f"disagreement_retry{retries}"
        )
        if retry_labels is not None:
            labels, content = retry_labels, retry_content

    result["n_disagreement_retries"] = retries
    result["final_labels"] = labels
    result["raw_reasoning"] = content
    result["model"] = calls[-1]["model"]

    if all(v == 1 for v in labels.values()):
        result["discarded"] = True
        result["discard_reason"] = "disagreement"
    else:
        result["correctness"] = 0
    return result


class BatchCounters:
    def __init__(self, prior_completed=0, prior_discarded=0, prior_parse_attempts=0, prior_parse_failures=0):
        self.lock = threading.Lock()
        self.n_completed = prior_completed
        self.n_discarded = prior_discarded
        self.parse_attempts = prior_parse_attempts
        self.parse_failures = prior_parse_failures
        self.checked_100 = prior_completed >= DISCARD_RATE_CHECK_N

    def record(self, result):
        calls = result["calls"]
        with self.lock:
            self.n_completed += 1
            if result["discarded"]:
                self.n_discarded += 1
            self.parse_attempts += len(calls)
            self.parse_failures += sum(1 for c in calls if c["parsed_ok"] is False)

            trigger = None
            if self.n_completed >= DISCARD_RATE_CHECK_N and not self.checked_100:
                self.checked_100 = True
                rate = self.n_discarded / self.n_completed
                if not (DISCARD_RATE_MIN <= rate <= DISCARD_RATE_MAX):
                    trigger = f"discard_rate_at_n={self.n_completed}_rate={rate:.4f}"
            if trigger is None and self.parse_attempts >= PARSE_FAIL_MIN_ATTEMPTS:
                fail_rate = self.parse_failures / self.parse_attempts
                if fail_rate > PARSE_FAIL_RATE_MAX:
                    trigger = f"parse_failure_rate={fail_rate:.4f}_n={self.parse_attempts}"
        return trigger


def worker(client, todo_iter, iter_lock, out_lock, state, counters, results_path):
    while not state.stop_event.is_set():
        with iter_lock:
            record = next(todo_iter, None)
        if record is None:
            return
        print(f"[start] record_id={record['record_id']} order={record['order_index']} bucket={record['bucket']}", flush=True)
        try:
            result = label_record(client, record, state)
        except FatalAPIError as e:
            print(f"[fatal] {e}", flush=True)
            return
        except TransientExhausted:
            with state.lock:
                state.records_failed_api.append(record["record_id"])
            print(f"[failed] record_id={record['record_id']} retry budget exhausted", flush=True)
            continue

        with out_lock:
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        trigger = counters.record(result)
        print(f"[done] record_id={record['record_id']} discarded={result['discarded']} n={counters.n_completed}", flush=True)
        if trigger:
            print(f"[guardrail] {trigger}", flush=True)
            state.request_stop(trigger)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=2500)
    parser.add_argument("--batch-name", default="batch1")
    args = parser.parse_args()

    prompt_sha8 = hashlib.sha256(PROMPT_FILE.read_bytes()).hexdigest()[:8]
    assert prompt_sha8 == EXPECTED_PROMPT_SHA8, f"prompt_4class_v3.py changed! sha8={prompt_sha8}"

    load_dotenv(DATA_DIR / ".env")
    api_key = os.environ["DEEPSEEK_API_KEY"]
    client = OpenAI(api_key=api_key, base_url=BASE_URL, max_retries=0, timeout=REQUEST_TIMEOUT)

    meta, candidates = load_candidates()
    print(f"Candidate pool: {meta}", flush=True)

    batch_slice = candidates[args.start:args.end]
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    results_jsonl = OUTPUTS_DIR / f"{args.batch_name}_results.jsonl"
    results_json = OUTPUTS_DIR / f"{args.batch_name}_results.json"
    summary_path = OUTPUTS_DIR / f"{args.batch_name}_summary.json"

    done = load_done_results(results_jsonl)
    done_ids = {r["record_id"] for r in done}
    todo = [r for r in batch_slice if r["record_id"] not in done_ids]
    print(f"Batch {args.batch_name} [{args.start}:{args.end}]: {len(batch_slice)} target, "
          f"{len(done_ids)} already done, {len(todo)} to run", flush=True)

    cfg = RunConfig("off", f"_{args.batch_name}_ds", {"thinking": {"type": "disabled"}},
                    1.0, True, None, avg_limit=AVG_TOKEN_LIMIT)
    prior_calls = sum(len(r["calls"]) for r in done)
    prior_tokens = sum(c["completion_tokens"] for r in done for c in r["calls"])
    state = RunState(cfg, prior_calls, prior_tokens)

    prior_discarded = sum(1 for r in done if r["discarded"])
    prior_parse_attempts = sum(len(r["calls"]) for r in done)
    prior_parse_failures = sum(1 for r in done for c in r["calls"] if c["parsed_ok"] is False)
    counters = BatchCounters(len(done), prior_discarded, prior_parse_attempts, prior_parse_failures)

    prev_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    balance_before = prev_summary.get("balance_before") or fetch_balance(api_key)
    started_at = prev_summary.get("started_at_utc") or utcnow()

    todo_iter = iter(todo)
    iter_lock, out_lock = threading.Lock(), threading.Lock()
    threads = [threading.Thread(target=worker, args=(client, todo_iter, iter_lock, out_lock, state, counters, results_jsonl))
               for _ in range(WORKERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    results_by_id = {r["record_id"]: r for r in load_done_results(results_jsonl)}
    ordered = [results_by_id[r["record_id"]] for r in batch_slice if r["record_id"] in results_by_id]
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=2, ensure_ascii=False)

    all_calls = [c for r in ordered for c in r["calls"]]
    error_counts = Counter(prev_summary.get("error_counts", {}))
    error_counts.update(state.error_counts)
    summary = {
        "batch_name": args.batch_name,
        "start": args.start,
        "end": args.end,
        "prompt_file": PROMPT_FILE.name,
        "prompt_sha256_prefix8": prompt_sha8,
        "candidate_pool_meta": meta,
        "thinking": "off",
        "temperature_sent": 1.0,
        "workers": WORKERS,
        "records_target": len(batch_slice),
        "records_done": len(ordered),
        "records_failed_api_retry_exhausted": sorted(
            set(prev_summary.get("records_failed_api_retry_exhausted", [])) | set(state.records_failed_api)),
        "stop_reason": state.stop_reason,
        "error_counts": dict(error_counts),
        "n_calls_logged": len(all_calls),
        "avg_completion_tokens_per_call": round(sum(c["completion_tokens"] for c in all_calls) / len(all_calls), 1) if all_calls else None,
        "avg_latency_s_per_call": round(sum(c["latency_s"] for c in all_calls) / len(all_calls), 1) if all_calls else None,
        "avg_output_token_limit": AVG_TOKEN_LIMIT,
        "discard_rate_check": {"n": DISCARD_RATE_CHECK_N, "range": [DISCARD_RATE_MIN, DISCARD_RATE_MAX]},
        "parse_failure_rate_limit": PARSE_FAIL_RATE_MAX,
        "balance_before": balance_before,
        "balance_after": fetch_balance(api_key),
        "started_at_utc": started_at,
        "ended_at_utc": utcnow(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
