"""Step 2: prior_ok=0 recheck for batch3 only, excluding triples already covered
by the first recheck round (recheck_prior_results_a/b.json, drawn from the
cumulative batch1+batch2 population).

Same two modes, same rules, same $-limit gate as recheck_prior.py; reuses its
helper functions directly. Writes recheck_prior_batch3_results_a/b.json and
recheck_prior_batch3_report.md (never touches the original recheck_prior_*
files).
"""
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from recheck_prior import (  # noqa: E402
    key_of, dedup_reps, run_mode, RunConfig, PRICE_IN, PRICE_OUT,
    EST_PROMPT_A, EST_COMPLETION_A, EST_PROMPT_B, EST_COMPLETION_B, COST_LIMIT_USD,
)

DATA_DIR = SCRIPT_DIR.parent
OUTPUTS_DIR = DATA_DIR / "outputs"


def main():
    batch1 = json.loads((OUTPUTS_DIR / "batch1_results.json").read_text(encoding="utf-8"))
    batch2 = json.loads((OUTPUTS_DIR / "batch2_results.json").read_text(encoding="utf-8"))
    batch3 = json.loads((OUTPUTS_DIR / "batch3_results.json").read_text(encoding="utf-8"))

    prior12 = json.loads((OUTPUTS_DIR / "recheck_prior_results_a.json").read_text(encoding="utf-8"))
    by_id_12 = {r["record_id"]: r for r in batch1 + batch2}
    already_rechecked_triples = {key_of(by_id_12[r["record_id"]]) for r in prior12}

    kept_prior0_b3 = [r for r in batch3 if not r["discarded"] and r["final_labels"]
                      and r["final_labels"]["prior_ok"] == 0]
    reps_all = dedup_reps(kept_prior0_b3)
    reps = [r for r in reps_all if key_of(r) not in already_rechecked_triples]
    n_skipped = len(reps_all) - len(reps)
    n = len(reps)

    cost_a = n * (EST_PROMPT_A * PRICE_IN + EST_COMPLETION_A * PRICE_OUT) / 1e6
    cost_b = n * (EST_PROMPT_B * PRICE_IN + EST_COMPLETION_B * PRICE_OUT) / 1e6
    est_total = cost_a + cost_b
    print(f"batch3 prior0 dedup groups: {len(reps_all)}, already covered by round-1 recheck: {n_skipped}, "
          f"new to recheck: {n}", flush=True)
    print(f"estimated cost: mode(a) ${cost_a:.3f}, mode(b) ${cost_b:.3f}, total ${est_total:.3f} "
          f"(limit ${COST_LIMIT_USD})", flush=True)

    estimate = {"n_groups_total": len(reps_all), "n_already_covered": n_skipped, "n_new": n,
                "cost_a_usd": round(cost_a, 4), "cost_b_usd": round(cost_b, 4),
                "total_usd": round(est_total, 4), "limit_usd": COST_LIMIT_USD}
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "recheck_prior_batch3_cost_estimate.json").write_text(
        json.dumps(estimate, indent=2, ensure_ascii=False), encoding="utf-8")

    if n == 0:
        (OUTPUTS_DIR / "recheck_prior_batch3_report.md").write_text(
            "# batch3 prior_ok=0 recheck - no new calls needed\n\n"
            f"batch3 has {len(reps_all)} deduplicated prior_ok=0 groups; all {n_skipped} of their "
            f"triples were already covered by the first recheck round (batch1+2), so no new calls "
            f"are needed.\n", encoding="utf-8")
        json.dump([], open(OUTPUTS_DIR / "recheck_prior_batch3_results_a.json", "w", encoding="utf-8"))
        json.dump([], open(OUTPUTS_DIR / "recheck_prior_batch3_results_b.json", "w", encoding="utf-8"))
        print("No new triples to recheck.", flush=True)
        return

    if est_total > COST_LIMIT_USD:
        (OUTPUTS_DIR / "recheck_prior_batch3_report.md").write_text(
            f"# batch3 prior_ok=0 recheck - paused, no API calls made\n\n"
            f"Estimated cost ${est_total:.3f} exceeds the configured limit ${COST_LIMIT_USD}; "
            f"no API calls were made.\n\n"
            f"New triples pending recheck: {n} ({n_skipped} groups already covered by the first "
            f"round and not repeated).\n",
            encoding="utf-8")
        print("PAUSED: estimated cost exceeds limit, no API calls made.", flush=True)
        return

    from dotenv import load_dotenv
    from openai import OpenAI
    import os
    load_dotenv(DATA_DIR / ".env")
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com",
                    max_retries=0, timeout=600)

    cfg_a = RunConfig("off", "_recheck_b3_a", {"thinking": {"type": "disabled"}}, 1.0, False, None)
    cfg_b = RunConfig("on", "_recheck_b3_b", None, None, False, None)

    results_a = run_mode(client, cfg_a, reps, "b3-a")
    (OUTPUTS_DIR / "recheck_prior_batch3_results_a.json").write_text(
        json.dumps(results_a, indent=2, ensure_ascii=False), encoding="utf-8")
    results_b = run_mode(client, cfg_b, reps, "b3-b")
    (OUTPUTS_DIR / "recheck_prior_batch3_results_b.json").write_text(
        json.dumps(results_b, indent=2, ensure_ascii=False), encoding="utf-8")

    a_by_id = {r["record_id"]: r for r in results_a}
    b_by_id = {r["record_id"]: r for r in results_b}
    a0 = sum(1 for r in reps if a_by_id[r["record_id"]]["labels"] and a_by_id[r["record_id"]]["labels"]["prior_ok"] == 0)
    b0 = sum(1 for r in reps if b_by_id[r["record_id"]]["labels"] and b_by_id[r["record_id"]]["labels"]["prior_ok"] == 0)

    def pct(x, n_):
        return f"{x}/{n_} = {x/n_*100:.1f}%" if n_ else "n/a"

    report = ["# batch3 prior_ok=0 recheck report", "",
              f"batch3 has {len(reps_all)} deduplicated prior_ok=0 groups. {n_skipped} of their "
              f"triples already appeared in the first recheck round (batch1+2, 158 groups) and "
              f"reuse those results directly instead of a new call. "
              f"{n} new calls were made this round.", "",
              f"(a) thinking off reproduction: {pct(a0, n)}", f"(b) thinking on reproduction: {pct(b0, n)}", ""]
    (OUTPUTS_DIR / "recheck_prior_batch3_report.md").write_text("\n".join(report), encoding="utf-8")

    actual_calls = [c for r in results_a + results_b for c in r["calls"]]
    actual_cost = sum((c["prompt_tokens"] * PRICE_IN + c["completion_tokens"] * PRICE_OUT) / 1e6 for c in actual_calls)
    print(f"Done. n_new={n}, actual cost=${actual_cost:.4f}", flush=True)


if __name__ == "__main__":
    main()
