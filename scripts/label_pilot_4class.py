"""Pilot run of the 4-class error-type labeling prompt on a small stratified sample
of PRM800K human_score(lt) == -1 records, using the Gemini API.

Reuses (structure only, not code) from the official PathFinder-PRM
src/data_generation/ scripts:
  - load_data.py: question + prev_steps + curr_step expansion
  - prompt.py: CLS_PROMPT context wrapping (see prompt_4class_v2.py for the 4-class variant)
  - run_inference.py: branching by human_score, and the disagreement-retry loop
    (max 2 retries; a record is discarded if every one of the 4 labels still says
    "no problem" after retries, since that contradicts the human lt=-1 annotation)

Scope: labeling pilot only. No training-format dataset is produced here.

v2 changes from v1 (prompt_4class_v1.py / pilot_4class_results_v1.json):
  - field names *_error/*_inconsistency -> *_ok (prompt_4class_v2.py)
  - Prior_ok narrowed to self-consistency with this solution's own prior steps
  - MODEL pinned to a concrete version string, not a "-latest" alias
  - temperature explicitly set (was implicit/unset in v1)
  - each record now also stores record_id (index into the full lt=-1 pool, for
    reproducibility checks) and model_version (as echoed back by the API)
  - kept (non-discarded) records get correctness=0 (lt=-1 records that survive
    the consistency filter are still human-labeled wrong overall)
"""
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from analyze_prm800k import load_prm800k_dataset, expand_records  # noqa: E402
from prompt_4class_v2 import CLS_PROMPT_4CLASS  # noqa: E402

DATA_DIR = SCRIPT_DIR.parent
OUTPUTS_DIR = DATA_DIR / "outputs"
RESULTS_SUFFIX = "_v2"
# concrete model gemini-flash-lite-latest currently resolves to (verified via
# resp.model_version); pinned per-instruction instead of using the alias
MODEL = "gemini-3.5-flash-lite"
TEMPERATURE = 1.0  # matches this model's documented default; set explicitly, not left implicit
SEED = 123
SAMPLES_PER_BUCKET = 30  # 3 buckets x 30 = 90, within the requested 50-100 range
MAX_PARSE_RETRIES = 3
MAX_DISAGREEMENT_RETRIES = 2  # matches official run_inference.py max_disagreement_retries
MAX_TRANSIENT_RETRIES = 5
TRANSIENT_BACKOFF_BASE = 5.0
SLEEP_BETWEEN_CALLS = 1.5

enc = tiktoken.get_encoding("cl100k_base")

LABEL_FIELDS = ["calc_ok", "method_ok", "prior_ok", "domain_ok"]
PARSE_RE = re.compile(
    r"Calc_ok:\s*([01])\s*"
    r"Method_ok:\s*([01])\s*"
    r"Prior_ok:\s*([01])\s*"
    r"Domain_ok:\s*([01])",
    re.IGNORECASE,
)


@dataclass
class RunStats:
    http_attempts: int = 0
    http_errors: int = 0
    count_503: int = 0
    count_429_transient: int = 0
    count_429_daily: int = 0
    parse_attempts: int = 0
    parse_failures: int = 0
    records_total: int = 0
    records_kept: int = 0
    records_discarded_unparseable: int = 0
    records_discarded_disagreement: int = 0
    records_with_disagreement_retry: int = 0
    model_versions_seen: set = None

    def __post_init__(self):
        if self.model_versions_seen is None:
            self.model_versions_seen = set()


def make_full_context(question_text, prev_steps, curr_step):
    # matches load_data.py's full_context f-string exactly
    return f"""
                [Math Problem]
                {question_text}
                [Partial Solution]
                <previous_steps>
                {prev_steps}
                </previous_steps>
                <current_step>
                {curr_step}
                </current_step>
                """


def stratified_sample(neg1_records):
    for r in neg1_records:
        r["tokens"] = len(enc.encode(r["question"] + r["prev_steps"] + r["curr_step"]))
    lengths = sorted(r["tokens"] for r in neg1_records)
    n = len(lengths)
    p33 = lengths[int(n * 0.33)]
    p67 = lengths[int(n * 0.67)]

    short = [r for r in neg1_records if r["tokens"] <= p33]
    medium = [r for r in neg1_records if p33 < r["tokens"] <= p67]
    long_ = [r for r in neg1_records if r["tokens"] > p67]

    rng = random.Random(SEED)
    sample = []
    for bucket_name, pool in [("short", short), ("medium", medium), ("long", long_)]:
        picked = rng.sample(pool, SAMPLES_PER_BUCKET)
        for r in picked:
            r = dict(r)
            r["bucket"] = bucket_name
            sample.append(r)
    rng.shuffle(sample)
    return sample, {"p33": p33, "p67": p67}


def parse_4class(text):
    if not text:
        return None
    matches = PARSE_RE.findall(text)
    if not matches:
        return None
    last = matches[-1]
    return {field_name: int(v) for field_name, v in zip(LABEL_FIELDS, last)}


class DailyQuotaExhausted(Exception):
    pass


def call_gemini_with_backoff(client, prompt, stats: RunStats):
    last_exc = None
    for attempt in range(MAX_TRANSIENT_RETRIES):
        stats.http_attempts += 1
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=genai_types.GenerateContentConfig(temperature=TEMPERATURE),
            )
            if resp.model_version:
                stats.model_versions_seen.add(resp.model_version)
            return resp.text, resp.model_version
        except genai_errors.APIError as e:
            stats.http_errors += 1
            is_daily = e.code == 429 and "PerDay" in str(e.details)
            if e.code == 503:
                stats.count_503 += 1
            elif e.code == 429 and is_daily:
                stats.count_429_daily += 1
            elif e.code == 429:
                stats.count_429_transient += 1
            # a per-day quota (as opposed to a per-minute rate limit) won't clear within
            # this run's retry budget, so fail fast instead of burning backoff time on it
            if is_daily:
                raise DailyQuotaExhausted(str(e)) from e
            last_exc = e
            wait = TRANSIENT_BACKOFF_BASE * (2 ** attempt)
            print(f"    [transient error, attempt {attempt+1}/{MAX_TRANSIENT_RETRIES}] {e}; sleeping {wait:.0f}s")
            time.sleep(wait)
    raise last_exc


def get_labels_with_parse_retry(client, prompt, stats: RunStats):
    raw_text = None
    model_version = None
    for attempt in range(MAX_PARSE_RETRIES):
        raw_text, model_version = call_gemini_with_backoff(client, prompt, stats)
        stats.parse_attempts += 1
        labels = parse_4class(raw_text)
        if labels is not None:
            return labels, raw_text, model_version
        stats.parse_failures += 1
        print(f"    [parse failure, attempt {attempt+1}/{MAX_PARSE_RETRIES}]")
        time.sleep(SLEEP_BETWEEN_CALLS)
    return None, raw_text, model_version


def label_record(client, record, stats: RunStats):
    full_context = make_full_context(record["question"], record["prev_steps"], record["curr_step"])
    prompt = CLS_PROMPT_4CLASS.format(context=full_context)

    labels, raw_text, model_version = get_labels_with_parse_retry(client, prompt, stats)

    result = {
        "record_id": record["record_id"],
        "bucket": record["bucket"],
        "tokens": record["tokens"],
        "question": record["question"],
        "prev_steps": record["prev_steps"],
        "curr_step": record["curr_step"],
        "human_score": record["human_score"],
        "final_labels": labels,
        "correctness": None,
        "raw_reasoning": raw_text,
        "model_version": model_version,
        "n_disagreement_retries": 0,
        "discarded": False,
        "discard_reason": None,
    }

    if labels is None:
        result["discarded"] = True
        result["discard_reason"] = "unparseable_after_retries"
        return result

    disagreement_retries = 0
    while all(v == 1 for v in labels.values()) and disagreement_retries < MAX_DISAGREEMENT_RETRIES:
        disagreement_retries += 1
        print(f"    disagreement (all-1) detected, retry {disagreement_retries}/{MAX_DISAGREEMENT_RETRIES}")
        retry_labels, retry_raw, retry_model_version = get_labels_with_parse_retry(client, prompt, stats)
        if retry_labels is not None:
            labels, raw_text, model_version = retry_labels, retry_raw, retry_model_version

    result["n_disagreement_retries"] = disagreement_retries
    result["final_labels"] = labels
    result["raw_reasoning"] = raw_text
    result["model_version"] = model_version

    if disagreement_retries > 0:
        stats.records_with_disagreement_retry += 1

    if all(v == 1 for v in labels.values()):
        result["discarded"] = True
        result["discard_reason"] = "persistent_disagreement"
    else:
        result["correctness"] = 0  # lt=-1 and passed the consistency filter: still human-labeled wrong overall

    return result


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    load_dotenv(DATA_DIR / ".env")
    api_key = os.environ["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)

    print("Loading PRM800K and expanding records...")
    dataset, _ = load_prm800k_dataset()
    records = expand_records(dataset)
    neg1_records = [dict(r, record_id=i) for i, r in enumerate(records) if r["human_score"] == -1]
    print(f"Total lt=-1 records available: {len(neg1_records)}")

    sample, cutoffs = stratified_sample(neg1_records)
    print(f"Stratified sample size: {len(sample)} (token tertile cutoffs: {cutoffs})")

    stats = RunStats(records_total=len(sample))
    results = []
    out_path = OUTPUTS_DIR / f"pilot_4class_results{RESULTS_SUFFIX}.json"

    quota_exhausted_msg = None
    for i, record in enumerate(sample):
        print(f"[{i+1}/{len(sample)}] record_id={record['record_id']} bucket={record['bucket']} tokens={record['tokens']}")
        try:
            result = label_record(client, record, stats)
        except DailyQuotaExhausted as e:
            quota_exhausted_msg = str(e)
            print(f"\nDaily free-tier quota exhausted after {len(results)}/{len(sample)} records. Stopping early.")
            break
        results.append(result)
        if result["discarded"]:
            if result["discard_reason"] == "persistent_disagreement":
                stats.records_discarded_disagreement += 1
            else:
                stats.records_discarded_unparseable += 1
        else:
            stats.records_kept += 1
        # checkpoint after every record so a later failure doesn't lose completed work
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        time.sleep(SLEEP_BETWEEN_CALLS)

    print(f"\nSaved {len(results)} labeled records to {out_path}")

    processed = len(results)
    summary = {
        "model_requested": MODEL,
        "model_versions_seen": sorted(stats.model_versions_seen),
        "temperature": TEMPERATURE,
        "sample_size_intended": stats.records_total,
        "records_processed": processed,
        "stopped_early_reason": "daily_quota_exhausted" if quota_exhausted_msg else None,
        "token_tertile_cutoffs": cutoffs,
        "http_attempts": stats.http_attempts,
        "http_errors": stats.http_errors,
        "count_503": stats.count_503,
        "count_429_transient": stats.count_429_transient,
        "count_429_daily": stats.count_429_daily,
        "http_success_rate": round(1 - stats.http_errors / stats.http_attempts, 4) if stats.http_attempts else None,
        "parse_attempts": stats.parse_attempts,
        "parse_failures": stats.parse_failures,
        "parse_failure_rate": round(stats.parse_failures / stats.parse_attempts, 4) if stats.parse_attempts else None,
        "records_kept": stats.records_kept,
        "records_discarded_unparseable": stats.records_discarded_unparseable,
        "records_discarded_disagreement": stats.records_discarded_disagreement,
        "disagreement_discard_rate": round(stats.records_discarded_disagreement / processed, 4) if processed else None,
        "records_with_disagreement_retry_triggered": stats.records_with_disagreement_retry,
    }
    summary_path = OUTPUTS_DIR / f"pilot_4class_summary{RESULTS_SUFFIX}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved summary to {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
