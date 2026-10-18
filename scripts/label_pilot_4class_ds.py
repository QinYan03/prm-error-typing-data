"""DeepSeek variant of the 4-class labeling pilot (prompt v2).

Same prompt, consistency filter and correctness rule as label_pilot_4class.py
(Gemini); only the model backend differs. Outputs carry a mode suffix so the
Gemini v1/v2 files are never touched.

Usage: label_pilot_4class_ds.py --thinking {on,off}
  off: whole frozen manifest, thinking disabled, temperature sent, halts when the
       average output tokens per call exceed AVG_OUTPUT_TOKEN_LIMIT or when
       reasoning tokens show up despite thinking being disabled.
  on:  API-default thinking on a supplement of Gemini-shared records, capped at
       SUPPLEMENT_MAX_CALLS requests per session.

The sample manifest is frozen on first creation. Runs are resumable: finished
record_ids in the results JSONL are skipped. Every mode also halts (results kept)
on insufficient balance or a non-retryable HTTP error.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
import openai
from dotenv import load_dotenv
from openai import OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from analyze_prm800k import load_prm800k_dataset, expand_records  # noqa: E402
from label_pilot_4class import (  # noqa: E402
    enc,
    make_full_context,
    parse_4class,
    MAX_PARSE_RETRIES,
    MAX_DISAGREEMENT_RETRIES,
)
from prompt_4class_v2 import CLS_PROMPT_4CLASS  # noqa: E402

DATA_DIR = SCRIPT_DIR.parent
OUTPUTS_DIR = DATA_DIR / "outputs"
PROMPT_FILE = SCRIPT_DIR / "prompt_4class_v2.py"
MODEL = "deepseek-flash"
BASE_URL = "https://api.deepseek.com"
WORKERS = 4
SEED_NEW = 456
P33, P67 = 188, 327
N_NEW_PER_BUCKET = {"short": 37, "medium": 37, "long": 36}
MAX_TRANSIENT_RETRIES = 5
BACKOFF_BASE = 3.0
REQUEST_TIMEOUT = 600
AVG_OUTPUT_TOKEN_LIMIT = 3000
MIN_CALLS_FOR_AVG_CHECK = 10

NOTHINK_TEMPERATURE = 1.0
SEED_SUPPLEMENT = 789
SUPPLEMENT_N = 21
SUPPLEMENT_MIN_SHARED_TOTAL = 25
SUPPLEMENT_MAX_CALLS = 60

MANIFEST_PATH = OUTPUTS_DIR / "pilot_4class_sample_ds.json"
SUPPLEMENT_PATH = OUTPUTS_DIR / "pilot_4class_sample_ds_think_supplement.json"


@dataclass
class RunConfig:
    thinking: str
    suffix: str
    extra_body: dict
    temperature: float
    avg_check: bool
    max_calls: int
    avg_limit: int = AVG_OUTPUT_TOKEN_LIMIT  # only read when avg_check is True

    @property
    def results_jsonl(self):
        return OUTPUTS_DIR / f"pilot_4class_results{self.suffix}.jsonl"

    @property
    def results_json(self):
        return OUTPUTS_DIR / f"pilot_4class_results{self.suffix}.json"

    @property
    def summary_path(self):
        return OUTPUTS_DIR / f"pilot_4class_summary{self.suffix}.json"


def make_config(thinking):
    if thinking == "off":
        return RunConfig("off", "_ds_nothink", {"thinking": {"type": "disabled"}},
                         NOTHINK_TEMPERATURE, True, None)
    return RunConfig("on", "_ds_think", None, None, False, SUPPLEMENT_MAX_CALLS)


class FatalAPIError(Exception):
    pass


class TransientExhausted(Exception):
    pass


def utcnow():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def bucket_of(tokens):
    if tokens <= P33:
        return "short"
    if tokens <= P67:
        return "medium"
    return "long"


def build_sample():
    dataset, _ = load_prm800k_dataset()
    records = expand_records(dataset)
    neg1 = [dict(r, record_id=i) for i, r in enumerate(records) if r["human_score"] == -1]
    for r in neg1:
        r["tokens"] = len(enc.encode(r["question"] + r["prev_steps"] + r["curr_step"]))
        r["bucket"] = bucket_of(r["tokens"])

    lengths = sorted(r["tokens"] for r in neg1)
    n = len(lengths)
    assert (lengths[int(n * 0.33)], lengths[int(n * 0.67)]) == (P33, P67), "tertile cutoffs drifted"
    by_id = {r["record_id"]: r for r in neg1}

    with open(OUTPUTS_DIR / "pilot_4class_results_v2.json", encoding="utf-8") as f:
        v2 = json.load(f)
    shared = []
    for v in v2:
        r = by_id[v["record_id"]]
        assert r["curr_step"] == v["curr_step"] and r["bucket"] == v["bucket"]
        shared.append(dict(r, source="v2_shared"))
    shared_ids = {r["record_id"] for r in shared}

    rng = random.Random(SEED_NEW)
    new = []
    for bucket, k in N_NEW_PER_BUCKET.items():
        pool = [r for r in neg1 if r["bucket"] == bucket and r["record_id"] not in shared_ids]
        new.extend(dict(r, source="new") for r in rng.sample(pool, k))

    sample = shared + new
    rng.shuffle(sample)
    return sample


def load_sample():
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            return json.load(f)
    sample = build_sample()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    return sample


class RunState:
    def __init__(self, cfg, prior_calls, prior_completion_tokens):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.stop_reason = None
        self.error_counts = Counter()
        self.records_failed_api = []
        self.n_calls = prior_calls
        self.sum_completion_tokens = prior_completion_tokens
        self.attempts_this_session = 0

    def count_error(self, key):
        with self.lock:
            self.error_counts[key] += 1

    def request_stop(self, reason):
        with self.lock:
            if self.stop_reason is None:
                self.stop_reason = reason
        self.stop_event.set()

    def reserve_attempt(self):
        """Count one outgoing request (retries included); refuse once the session cap is spent."""
        with self.lock:
            if self.cfg.max_calls is not None and self.attempts_this_session >= self.cfg.max_calls:
                capped = True
            else:
                capped = False
                self.attempts_this_session += 1
        if capped:
            self.request_stop("max_calls_reached")
            raise FatalAPIError("session call cap reached")

    def record_call(self, completion_tokens):
        with self.lock:
            self.n_calls += 1
            self.sum_completion_tokens += completion_tokens
            avg = self.sum_completion_tokens / self.n_calls
            exceeded = (self.cfg.avg_check and self.n_calls >= MIN_CALLS_FOR_AVG_CHECK
                        and avg > self.cfg.avg_limit)
        if exceeded:
            self.request_stop("avg_output_tokens_exceeded")


def call_deepseek(client, prompt, state, phase):
    """One chat completion with backoff on 429/5xx/connection errors.

    Raises FatalAPIError on 402 and other non-retryable statuses (run halts),
    TransientExhausted when the retry budget is spent.
    """
    cfg = state.cfg
    request_kwargs = {}
    if cfg.extra_body:
        request_kwargs["extra_body"] = cfg.extra_body
    if cfg.temperature is not None:
        request_kwargs["temperature"] = cfg.temperature

    for attempt in range(MAX_TRANSIENT_RETRIES):
        state.reserve_attempt()
        started_at = utcnow()
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": prompt}], **request_kwargs
            )
        except openai.APIStatusError as e:
            code = e.status_code
            state.count_error(str(code))
            if code == 402:
                state.request_stop("insufficient_balance")
                raise FatalAPIError("402 insufficient balance") from e
            if code in (400, 401, 422):
                state.request_stop(f"fatal_http_{code}")
                raise FatalAPIError(f"HTTP {code}: {e}") from e
            wait = BACKOFF_BASE * (2 ** attempt)
            print(f"    [HTTP {code}, attempt {attempt+1}/{MAX_TRANSIENT_RETRIES}] retry in {wait:.0f}s", flush=True)
            time.sleep(wait)
            continue
        except openai.APITimeoutError:
            state.count_error("timeout")
        except openai.APIConnectionError:
            state.count_error("connection")
        else:
            usage = resp.usage
            details = usage.completion_tokens_details
            msg = resp.choices[0].message
            call = {
                "phase": phase,
                "started_at_utc": started_at,
                "ended_at_utc": utcnow(),
                "latency_s": round(time.time() - t0, 2),
                "model": resp.model,
                "finish_reason": resp.choices[0].finish_reason,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "reasoning_tokens": (details.reasoning_tokens if details else None),
                "cache_hit_tokens": getattr(usage, "prompt_cache_hit_tokens", None),
                "cache_miss_tokens": getattr(usage, "prompt_cache_miss_tokens", None),
                "reasoning_content_chars": len(getattr(msg, "reasoning_content", None) or ""),
                "thinking": cfg.thinking,
                "temperature_sent": cfg.temperature,
                "parsed_ok": None,
            }
            state.record_call(usage.completion_tokens)
            if cfg.thinking == "off" and ((call["reasoning_tokens"] or 0) > 0 or call["reasoning_content_chars"] > 0):
                print(f"[fatal] reasoning present with thinking disabled: {call}", flush=True)
                state.request_stop("thinking_not_disabled")
                raise FatalAPIError("thinking not disabled")
            return msg.content, call
        wait = BACKOFF_BASE * (2 ** attempt)
        print(f"    [network error, attempt {attempt+1}/{MAX_TRANSIENT_RETRIES}] retry in {wait:.0f}s", flush=True)
        time.sleep(wait)
    raise TransientExhausted()


def get_labels_with_parse_retry(client, prompt, state, calls, phase):
    content = None
    for attempt in range(MAX_PARSE_RETRIES):
        call_phase = phase if attempt == 0 else f"{phase}/parse_retry{attempt}"
        content, call = call_deepseek(client, prompt, state, call_phase)
        labels = parse_4class(content)
        call["parsed_ok"] = labels is not None
        calls.append(call)
        if labels is not None:
            return labels, content
    return None, content


def label_record(client, record, state):
    full_context = make_full_context(record["question"], record["prev_steps"], record["curr_step"])
    prompt = CLS_PROMPT_4CLASS.format(context=full_context)
    calls = []

    labels, content = get_labels_with_parse_retry(client, prompt, state, calls, "initial")

    result = {
        "record_id": record["record_id"],
        "source": record["source"],
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
        result["discard_reason"] = "unparseable_after_retries"
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
        result["discard_reason"] = "persistent_disagreement"
    else:
        result["correctness"] = 0
    return result


def worker(client, todo_iter, iter_lock, out_lock, state):
    while not state.stop_event.is_set():
        with iter_lock:
            record = next(todo_iter, None)
        if record is None:
            return
        print(f"[start] record_id={record['record_id']} bucket={record['bucket']} tokens={record['tokens']}", flush=True)
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
            with open(state.cfg.results_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"[done] record_id={record['record_id']} discarded={result['discarded']} calls={len(result['calls'])}", flush=True)


def load_done_results(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def select_todo(cfg, sample, done_ids):
    if cfg.thinking == "off":
        return [r for r in sample if r["record_id"] not in done_ids]

    if SUPPLEMENT_PATH.exists():
        chosen_ids = json.loads(SUPPLEMENT_PATH.read_text(encoding="utf-8"))["record_ids"]
    else:
        shared_done = sum(1 for r in sample if r["record_id"] in done_ids and r["source"] == "v2_shared")
        assert shared_done + SUPPLEMENT_N >= SUPPLEMENT_MIN_SHARED_TOTAL
        pool = sorted(r["record_id"] for r in sample if r["source"] == "v2_shared" and r["record_id"] not in done_ids)
        chosen_ids = random.Random(SEED_SUPPLEMENT).sample(pool, SUPPLEMENT_N)
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        SUPPLEMENT_PATH.write_text(
            json.dumps({"seed": SEED_SUPPLEMENT, "record_ids": chosen_ids}, indent=2), encoding="utf-8")
    by_id = {r["record_id"]: r for r in sample}
    return [by_id[i] for i in chosen_ids if i not in done_ids]


def fetch_balance(api_key):
    try:
        r = httpx.get(f"{BASE_URL}/user/balance", headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
        return r.json().get("balance_infos") if r.status_code == 200 else None
    except httpx.HTTPError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thinking", choices=["on", "off"], required=True)
    cfg = make_config(parser.parse_args().thinking)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    load_dotenv(DATA_DIR / ".env")
    api_key = os.environ["DEEPSEEK_API_KEY"]
    client = OpenAI(api_key=api_key, base_url=BASE_URL, max_retries=0, timeout=REQUEST_TIMEOUT)

    sample = load_sample()
    done = load_done_results(cfg.results_jsonl)
    done_ids = {r["record_id"] for r in done}
    todo = select_todo(cfg, sample, done_ids)
    print(f"Mode thinking={cfg.thinking}: sample {len(sample)}, already done: {len(done_ids)}, to run: {len(todo)}", flush=True)

    prior_calls = sum(len(r["calls"]) for r in done)
    prior_tokens = sum(c["completion_tokens"] for r in done for c in r["calls"])
    state = RunState(cfg, prior_calls, prior_tokens)

    prev_summary = json.loads(cfg.summary_path.read_text()) if cfg.summary_path.exists() else {}
    balance_before = prev_summary.get("balance_before") or fetch_balance(api_key)
    started_at = prev_summary.get("started_at_utc") or utcnow()

    todo_iter = iter(todo)
    iter_lock, out_lock = threading.Lock(), threading.Lock()
    threads = [
        threading.Thread(target=worker, args=(client, todo_iter, iter_lock, out_lock, state))
        for _ in range(WORKERS)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    results_by_id = {r["record_id"]: r for r in load_done_results(cfg.results_jsonl)}
    ordered = [results_by_id[r["record_id"]] for r in sample if r["record_id"] in results_by_id]
    with open(cfg.results_json, "w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=2, ensure_ascii=False)

    error_counts = Counter(prev_summary.get("error_counts", {}))
    error_counts.update(state.error_counts)
    all_calls = [c for r in ordered for c in r["calls"]]
    summary = {
        "model_requested": MODEL,
        "model_field_values": dict(Counter(c["model"] for c in all_calls)),
        "thinking": cfg.thinking,
        "temperature_sent": cfg.temperature,
        "thinking_settings": (
            'extra_body={"thinking":{"type":"disabled"}}' if cfg.thinking == "off"
            else "no thinking/reasoning_effort params sent; API default applies "
                 "(documented default: thinking enabled, effort high)"
        ),
        "prompt_file": PROMPT_FILE.name,
        "prompt_sha256": hashlib.sha256(PROMPT_FILE.read_bytes()).hexdigest(),
        "max_calls_this_session": cfg.max_calls,
        "attempts_this_session": state.attempts_this_session,
        "workers": WORKERS,
        "sample_size_planned": len(sample),
        "records_done": len(ordered),
        "records_failed_api_retry_exhausted": sorted(set(prev_summary.get("records_failed_api_retry_exhausted", [])) | set(state.records_failed_api)),
        "stop_reason": state.stop_reason,
        "error_counts": dict(error_counts),
        "n_calls_logged": len(all_calls),
        "avg_completion_tokens_per_call": round(sum(c["completion_tokens"] for c in all_calls) / len(all_calls), 1) if all_calls else None,
        "avg_latency_s_per_call": round(sum(c["latency_s"] for c in all_calls) / len(all_calls), 1) if all_calls else None,
        "avg_output_token_limit": cfg.avg_limit if cfg.avg_check else None,
        "balance_before": balance_before,
        "balance_after": fetch_balance(api_key),
        "started_at_utc": started_at,
        "ended_at_utc": utcnow(),
    }
    with open(cfg.summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
