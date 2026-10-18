"""format_preview.md: worked examples of the 4 training-format versions (0/1/2/4
class) plus their system prompts, on 4 hand-picked selection_v1.json records.
No API calls, no full dataset built. String construction mirrors the official
build_dataset.py / build_dataset_ablation.py exactly (see those files for the
GOOD_TOKEN/BAD_TOKEN/SEPARATOR_TOKEN and asst_prompt/build_conversation logic).
"""
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from analyze_prm800k import load_prm800k_dataset, expand_records  # noqa: E402

DATA_DIR = SCRIPT_DIR.parent
OUTPUTS_DIR = DATA_DIR / "outputs"
GOOD_TOKEN, BAD_TOKEN, SEPARATOR_TOKEN = "<+>", "<->", "<extra>"

SYSTEM_PROMPTS = {
    "0class": "You are a Math Teacher. Given a question and a student's solution, "
              "evaluate whether the current step will lead to the correct final solution",
    "1class": "You are a Math Teacher. Given a question and a student's solution, "
              "evaluate whether the current step contains any error and whether it "
              "will lead to the correct final solution",
    "2class": "You are a Math Teacher. Given a question and a student's solution, evaluate the "
              "mathematical correctness, logic consistency of the current step and whether "
              "it will lead to the correct final solution",
    "4class": "You are a Math Teacher. Given a question and a student's solution, evaluate the "
              "calculation correctness, method validity, consistency with prior steps, and "
              "consistency with the problem domain of the current step and whether it will "
              "lead to the correct final solution",
}


def tok(v):
    return GOOD_TOKEN if v == 1 else BAD_TOKEN


def asst_prefix(prev_steps, curr_step):
    if len(prev_steps) > 0:
        return prev_steps + "\n\nCurrent Step: " + curr_step
    return "Current Step: " + curr_step


def user_content(version, question):
    return SYSTEM_PROMPTS[version] + "\n\n Question: " + question


def build_conversation_row(version, stage, question, label_asst, input_asst):
    return {
        "labels": [
            {"role": "user", "content": user_content(version, question)},
            {"role": "assistant", "content": label_asst},
        ],
        "inputs": [
            {"role": "user", "content": user_content(version, question)},
            {"role": "assistant", "content": input_asst},
        ],
    }


def labels_for(rec):
    """rec: {'labels': {calc_ok,method_ok,prior_ok,domain_ok}, 'human_score': int}
    Correct/suboptimal side (human_score != -1) is always treated as all-ok,
    since these records never go through the 4-class labeler."""
    if rec["human_score"] == -1:
        lab = rec["labels"]
    else:
        lab = {"calc_ok": 1, "method_ok": 1, "prior_ok": 1, "domain_ok": 1}
    correctness = 1 if rec["human_score"] == 1 else 0  # lt=0 and lt=-1 both -> 0
    math_ok = 1 if (lab["calc_ok"] == 1 and lab["method_ok"] == 1) else 0
    consistency_ok = 1 if (lab["prior_ok"] == 1 and lab["domain_ok"] == 1) else 0
    error_ok = 1 if all(v == 1 for v in lab.values()) else 0
    return lab, correctness, math_ok, consistency_ok, error_ok


def build_all_versions(rec, meta):
    asst = asst_prefix(rec["prev_steps"], rec["curr_step"])
    lab, correctness, math_ok, consistency_ok, error_ok = labels_for(rec)
    c_tok = tok(correctness)
    rows = {}

    # 0class: single stage, ablation-style
    lbl = f" Correctness: {c_tok}"
    inp = f" Correctness: {SEPARATOR_TOKEN}"
    row = build_conversation_row("0class", "single", rec["question"], asst + lbl, asst + inp)
    row.update({"id": f"{meta['record_id']}_0class_single", "record_id": meta["record_id"],
               "question_id": meta["question_id"], "side": meta["side"],
               "human_score": rec["human_score"], "version": "0class", "stage": "single"})
    rows["0class"] = [row]

    # 1class: two stage, Error label (paper's "w/o Separate Error Categories" ablation,
    # no official script -- our own implementation)
    e_tok = tok(error_ok)
    s1_lbl = f" Error: {e_tok}"
    s1_inp = f" Error: {SEPARATOR_TOKEN}"
    s2_lbl = s1_lbl + f", Correctness: {c_tok}"
    s2_inp = s1_lbl + f", Correctness: {SEPARATOR_TOKEN}"
    r1 = build_conversation_row("1class", 1, rec["question"], asst + s1_lbl, asst + s1_inp)
    r1.update({"id": f"{meta['record_id']}_1class_1", "record_id": meta["record_id"],
              "question_id": meta["question_id"], "side": meta["side"],
              "human_score": rec["human_score"], "version": "1class", "stage": 1})
    r2 = build_conversation_row("1class", 2, rec["question"], asst + s2_lbl, asst + s2_inp)
    r2.update({"id": f"{meta['record_id']}_1class_2", "record_id": meta["record_id"],
              "question_id": meta["question_id"], "side": meta["side"],
              "human_score": rec["human_score"], "version": "1class", "stage": 2})
    rows["1class"] = [r1, r2]

    # 2class: two stage, matches official build_dataset.py verbatim
    m_tok, cons_tok = tok(math_ok), tok(consistency_ok)
    s1_lbl = f" Math reasoning: {m_tok}, Consistency: {cons_tok}"
    s1_inp = f" Math reasoning: {SEPARATOR_TOKEN}, Consistency: {SEPARATOR_TOKEN}"
    s2_lbl = s1_lbl + f", Correctness: {c_tok}"
    s2_inp = s1_lbl + f", Correctness: {SEPARATOR_TOKEN}"
    r1 = build_conversation_row("2class", 1, rec["question"], asst + s1_lbl, asst + s1_inp)
    r1.update({"id": f"{meta['record_id']}_2class_1", "record_id": meta["record_id"],
              "question_id": meta["question_id"], "side": meta["side"],
              "human_score": rec["human_score"], "version": "2class", "stage": 1})
    r2 = build_conversation_row("2class", 2, rec["question"], asst + s2_lbl, asst + s2_inp)
    r2.update({"id": f"{meta['record_id']}_2class_2", "record_id": meta["record_id"],
              "question_id": meta["question_id"], "side": meta["side"],
              "human_score": rec["human_score"], "version": "2class", "stage": 2})
    rows["2class"] = [r1, r2]

    # 4class: two stage, our extension
    calc_t, method_t, prior_t, domain_t = (tok(lab["calc_ok"]), tok(lab["method_ok"]),
                                           tok(lab["prior_ok"]), tok(lab["domain_ok"]))
    s1_lbl = (f" Calculation: {calc_t}, Method: {method_t}, "
             f"Prior-step consistency: {prior_t}, Domain consistency: {domain_t}")
    s1_inp = (f" Calculation: {SEPARATOR_TOKEN}, Method: {SEPARATOR_TOKEN}, "
             f"Prior-step consistency: {SEPARATOR_TOKEN}, Domain consistency: {SEPARATOR_TOKEN}")
    s2_lbl = s1_lbl + f", Correctness: {c_tok}"
    s2_inp = s1_lbl + f", Correctness: {SEPARATOR_TOKEN}"
    r1 = build_conversation_row("4class", 1, rec["question"], asst + s1_lbl, asst + s1_inp)
    r1.update({"id": f"{meta['record_id']}_4class_1", "record_id": meta["record_id"],
              "question_id": meta["question_id"], "side": meta["side"],
              "human_score": rec["human_score"], "version": "4class", "stage": 1})
    r2 = build_conversation_row("4class", 2, rec["question"], asst + s2_lbl, asst + s2_inp)
    r2.update({"id": f"{meta['record_id']}_4class_2", "record_id": meta["record_id"],
              "question_id": meta["question_id"], "side": meta["side"],
              "human_score": rec["human_score"], "version": "4class", "stage": 2})
    rows["4class"] = [r1, r2]

    return rows, {"raw_labels": lab, "correctness": correctness, "math_ok": math_ok,
                 "consistency_ok": consistency_ok, "error_ok": error_ok}


def pick_examples():
    sel = json.loads((OUTPUTS_DIR / "selection_v1.json").read_text(encoding="utf-8"))["records"]
    confirmed = {r["record_id"]: r for r in json.loads((OUTPUTS_DIR / "labels_confirmed_v1.json")
                                                        .read_text(encoding="utf-8"))}
    dataset, _ = load_prm800k_dataset()
    expanded = expand_records(dataset)

    prior_example = next(r for r in sel if r["side"] == "error" and r["confirmed_labels"]["prior_ok"] == 0)
    calc_only_example = next(
        r for r in sel if r["side"] == "error" and r["confirmed_labels"]["calc_ok"] == 0
        and r["confirmed_labels"]["method_ok"] == 1 and r["confirmed_labels"]["prior_ok"] == 1
        and r["confirmed_labels"]["domain_ok"] == 1
    )
    lt1_example = next(r for r in sel if r["side"] == "correct" and r["human_score"] == 1)
    lt0_example = next(r for r in sel if r["side"] == "correct" and r["human_score"] == 0)

    out = []
    for tag, ex in [("error-prior_ok=0", prior_example), ("error-calc_ok=0-only", calc_only_example),
                    ("correct-lt=1", lt1_example), ("correct-lt=0", lt0_example)]:
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
        out.append((tag, rec, {"record_id": rid, "question_id": ex["question_id"], "side": ex["side"]}))
    return out


def diff_system_prompts():
    base = SYSTEM_PROMPTS["2class"]
    lines = []
    for v in ["0class", "1class", "4class"]:
        cur = SYSTEM_PROMPTS[v]
        lines.append(f"### {v} vs 2class (official original)")
        lines.append("")
        lines.append(f"- Official original: `{base}`")
        lines.append(f"- {v}: `{cur}`")
        # locate the changed clause
        prefix = "You are a Math Teacher. Given a question and a student's solution, evaluate "
        suffix = " of the current step and whether it will lead to the correct final solution"
        base_mid = base[len(prefix):-len(suffix)] if base.endswith(suffix) else None
        if v == "1class":
            cur_mid = "whether the current step contains any error"
            lines.append(f"- Difference: `{base_mid} of the current step` is replaced in full by "
                         f"`{cur_mid}` (prefix and suffix unchanged).")
        elif v == "4class":
            cur_mid = ("the calculation correctness, method validity, consistency with prior "
                      "steps, and consistency with the problem domain")
            lines.append(f"- Difference: `{base_mid}` is replaced by `{cur_mid}` (rest unchanged).")
        elif v == "0class":
            lines.append(f"- Difference: `{base_mid} of the current step and` is removed entirely "
                         f"(0-class has no separate error category to evaluate, so only the "
                         f"lead-to-correct-answer clause remains; prefix and suffix unchanged).")
        lines.append("")
    return "\n".join(lines)


def main():
    examples = pick_examples()

    L = ["# Training format sample preview (format_preview.md)", "",
         "Only expands 4 hand-picked examples; the full dataset is not built and no API is called. "
         "The string construction mirrors the official `build_dataset.py` (2-class, two-stage) and "
         "`build_dataset_ablation.py` (0-class, single-stage) exactly; 1-class and 4-class reuse the "
         "same concatenation structure with different label strings.", ""]

    L.append("## Step 1: full JSONL rows for 4 examples x 4 versions")
    L.append("")
    for tag, rec, meta in examples:
        rows, agg = build_all_versions(rec, meta)
        L.append(f"### Example: {tag} (record_id={meta['record_id']}, side={meta['side']}, "
                 f"human_score={rec['human_score']})")
        L.append("")
        L.append(f"Raw label vector: {agg['raw_labels']} (correct-side records are treated as "
                 f"all-ok by rule, not an annotation result)")
        L.append(f"Per-version aggregated result: Correctness={agg['correctness']}, "
                 f"2-class Math_ok={agg['math_ok']}, "
                 f"2-class Consistency_ok={agg['consistency_ok']}, 1-class Error_ok={agg['error_ok']}")
        L.append("")
        for version in ["0class", "1class", "2class", "4class"]:
            L.append(f"**{version}:**")
            L.append("")
            for row in rows[version]:
                L.append("```json")
                L.append(json.dumps(row, ensure_ascii=False, indent=2))
                L.append("```")
            L.append("")

    L.append("## Step 2: full text of the four system prompts + differences")
    L.append("")
    for v in ["0class", "1class", "2class", "4class"]:
        note = "(official build_dataset.py original, unchanged)" if v == "2class" else "(see the diff notes below)"
        L.append(f"**{v}** {note}:")
        L.append("")
        L.append(f"> {SYSTEM_PROMPTS[v]}")
        L.append("")
    L.append(diff_system_prompts())

    L.append("## Step 3: projected full row counts per version")
    L.append("")
    L.append("| Version | Rows per record | Projected full count (error side 1165 + correct side 1165 = 2330 records) |")
    L.append("|---|---:|---:|")
    L.append("| 0class | 1 (single stage) | 2330 |")
    L.append("| 1class | 2 (two stage) | 4660 |")
    L.append("| 2class | 2 (two stage) | 4660 |")
    L.append("| 4class | 2 (two stage) | 4660 |")
    L.append("")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "format_preview.md").write_text("\n".join(L), encoding="utf-8")
    print("Wrote format_preview.md")


if __name__ == "__main__":
    main()
