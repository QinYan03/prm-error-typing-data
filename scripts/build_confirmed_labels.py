"""Step 3: labels_confirmed_v1.json over the cumulative batch1+2+3 (7500) labels.

Confirmation rule: a kept record's prior_ok=0 is confirmed
(stays 0) only if, for its (question, prev_steps, curr_step) triple, at least
one of the two independent recheck modes (a: thinking off resample, b: thinking
on) also gave prior_ok=0 for that triple's rechecked representative. Otherwise
prior_ok is flipped to 1 in confirmed_labels. If flipping makes all four
confirmed labels 1, the record is marked discarded with
discard_reason="prior_unconfirmed" (correctness becomes None, matching how
other discarded records are represented) and no longer counts as kept.

Original labels/discard status are untouched (kept as original_labels /
original_discarded / original_discard_reason); this file only adds the
confirmed_* fields alongside them.

Triples with no recheck coverage at all (should not occur for prior_ok=0 kept
records given how recheck candidates were selected, but checked defensively)
are left unconfirmed (kept as-is) and counted separately as a caveat.
"""
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from recheck_prior import key_of  # noqa: E402

DATA_DIR = SCRIPT_DIR.parent
OUTPUTS_DIR = DATA_DIR / "outputs"


def load(name):
    return json.loads((OUTPUTS_DIR / name).read_text(encoding="utf-8"))


def build_confirmed_prior0_map():
    """triple -> True/False (any-mode recheck said prior_ok==0), across all
    recheck rounds run so far (batch1+2's 158 groups, plus batch3's new ones)."""
    batch1, batch2, batch3 = load("batch1_results.json"), load("batch2_results.json"), load("batch3_results.json")
    by_id_12 = {r["record_id"]: r for r in batch1 + batch2}
    by_id_3 = {r["record_id"]: r for r in batch3}

    a1 = load("recheck_prior_results_a.json")
    b1 = load("recheck_prior_results_b.json")
    a3 = load("recheck_prior_batch3_results_a.json") if (OUTPUTS_DIR / "recheck_prior_batch3_results_a.json").exists() else []
    b3 = load("recheck_prior_batch3_results_b.json") if (OUTPUTS_DIR / "recheck_prior_batch3_results_b.json").exists() else []

    triple_map = {}

    def ingest(records_a, records_b, by_id):
        b_by_id = {r["record_id"]: r for r in records_b}
        for ra in records_a:
            rid = ra["record_id"]
            triple = key_of(by_id[rid])
            rb = b_by_id.get(rid)
            a0 = bool(ra["labels"] and ra["labels"]["prior_ok"] == 0)
            b0 = bool(rb and rb["labels"] and rb["labels"]["prior_ok"] == 0)
            triple_map[triple] = a0 or b0

    ingest(a1, b1, by_id_12)
    ingest(a3, b3, by_id_3)
    return triple_map


def confirm_record(r, triple_map, stats):
    orig = r["final_labels"]
    confirmed = dict(orig) if orig else None
    confirmed_discarded = r["discarded"]
    confirmed_discard_reason = r["discard_reason"]
    confirmed_correctness = r["correctness"]

    if orig and orig["prior_ok"] == 0 and not r["discarded"]:
        triple = key_of(r)
        if triple in triple_map:
            if not triple_map[triple]:
                confirmed["prior_ok"] = 1
                stats["flipped"] += 1
                if all(v == 1 for v in confirmed.values()):
                    confirmed_discarded = True
                    confirmed_discard_reason = "prior_unconfirmed"
                    confirmed_correctness = None
                    stats["newly_discarded"] += 1
        else:
            stats["no_recheck_coverage"] += 1

    return {
        "record_id": r["record_id"],
        "bucket": r["bucket"],
        "tokens": r["tokens"],
        "question": r["question"],
        "prev_steps": r["prev_steps"],
        "curr_step": r["curr_step"],
        "human_score": r["human_score"],
        "original_labels": orig,
        "original_discarded": r["discarded"],
        "original_discard_reason": r["discard_reason"],
        "confirmed_labels": confirmed,
        "confirmed_discarded": confirmed_discarded,
        "confirmed_discard_reason": confirmed_discard_reason,
        "confirmed_correctness": confirmed_correctness,
    }


def main():
    batch1, batch2, batch3 = load("batch1_results.json"), load("batch2_results.json"), load("batch3_results.json")
    cumulative = batch1 + batch2 + batch3
    triple_map = build_confirmed_prior0_map()

    stats = {"flipped": 0, "newly_discarded": 0, "no_recheck_coverage": 0}
    out = [confirm_record(r, triple_map, stats) for r in cumulative]

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "labels_confirmed_v1.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote labels_confirmed_v1.json: {len(out)} records")
    print(f"prior_ok flipped 0->1: {stats['flipped']}, newly discarded (prior_unconfirmed): "
          f"{stats['newly_discarded']}, triples with no recheck coverage: {stats['no_recheck_coverage']}")


if __name__ == "__main__":
    main()
