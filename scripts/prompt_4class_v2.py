"""4-way error-type labeling prompt, v2.

Derived from the official PathFinder-PRM src/data_generation/prompt.py CLS_PROMPT.
Criterion A (Mathematical logic) is split into Calc_ok / Method_ok.
Criterion B (Consistency) is split into Prior_ok / Domain_ok.
Criterion C (Simplicity and optimality) is dropped: it is not an error-type label,
it maps to the separate downstream "correctness" field, which is assigned outside
this prompt.

Score convention: 0 = problem present, 1 = no problem.

Changes from v1 (prompt_4class_v1.py):
- Field names renamed *_error/*_inconsistency -> *_ok (same 0/1 direction).
- Prior_ok narrowed to self-consistency with this solution's own previous steps
  only; consistency with the problem statement's given conditions is Domain_ok's
  job instead (v1 let "constraints established earlier" bleed into both).
- Prior_ok is defined to default to 1 when there are no previous steps.
"""

CLS_PROMPT_4CLASS = """
You are an analytical math instructor grading a student's work. Think step-by-step through your analysis. Below is the math question, previous steps by the student, and current step to evaluate.

{context}

Your task is to rigorously examine the current step and determine if it contains ANY mathematical errors. Assign binary scores (0 = problem present, 1 = no problem) based on four criteria:

Calc_ok) Calculation — Is the current step, **on its own**, free of arithmetic/algebraic mistakes? Check for:
   • Calculation errors
   • Invalid operations or simplifications
   • Algebra mistakes or sign errors

Method_ok) Method/formula — Is the current step, **on its own**, using a valid approach or formula? Check for:
   • Incorrect formula application
   • Incorrect assertions
   • Using the wrong method/theorem for this situation

Prior_ok) Consistency with this solution's own previous steps — Is the current step consistent with the conclusions and assumptions that **this solution itself** has already derived or established in its own previous steps? This is about self-consistency with the solution-so-far, **not** about the conditions given in the math problem statement — those belong under Domain_ok. If there are no previous steps yet, this criterion is automatically satisfied (score 1).

Domain_ok) Consistency with problem/domain — Is the current step consistent with:
   • The conditions and ground truth given in the math problem statement itself
   • The mathematical domain applicable to this problem (e.g. domain restrictions on variables, applicability conditions of a theorem)

Double check all four criteria explicitly in your reasoning. In your analysis, be sensitive to subtle issues like missing pre-requisites/assumptions, correct-looking statements with slight errors and high confidence statements containing errors.

IMPORTANT POINTS:

- If you find ANY error, even a minor one, you MUST assign a score of 0 to the appropriate criterion. Be skeptical and verify all claims thoroughly.

- For incorrect steps, wherever possible, attempt to categorize the issue as violating **one of the four criteria** (i.e., assign score 0 to **only one category**). Assign multiple 0 scores only for serious errors that genuinely span more than one category.

You must format your answer as below:

Reasoning:
{{Provide detailed analysis, showing all verification steps and explicitly identifying any errors found}}

Final answers:
Calc_ok:
{{0 or 1 only}}
Method_ok:
{{0 or 1 only}}
Prior_ok:
{{0 or 1 only}}
Domain_ok:
{{0 or 1 only}}


"""
