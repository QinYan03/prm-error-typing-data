"""4-way error-type labeling prompt, v3.

Same structure, output fields (Calc_ok/Method_ok/Prior_ok/Domain_ok) and 0/1
direction (0 = problem present, 1 = no problem) as v2 (prompt_4class_v2.py).
The four categories are now mapped bullet-by-bullet onto the original paper's
Appendix D criteria A/B, plus one proposal-only addition to Domain_ok.
"""

CLS_PROMPT_4CLASS = """
You are an analytical math instructor grading a student's work. Think step-by-step through your analysis. Below is the math question, previous steps by the student, and current step to evaluate.

{context}

Your task is to rigorously examine the current step and determine if it contains ANY mathematical errors. Assign binary scores (0 = problem present, 1 = no problem) based on four criteria:

Calc_ok) Calculation — Is the current step, **on its own**, free of arithmetic/algebraic mistakes? Check for:
   • Calculation errors
   • Algebra mistakes or sign errors
   (The operation, formula or approach chosen is assumed valid here; only whether it was executed correctly is in scope. A valid operation that is merely miscalculated is a Calc_ok problem.)

Method_ok) Method/formula — Is the current step, **on its own**, using a valid approach or formula? Check for:
   • Incorrect formula application
   • Invalid operations or simplifications
   • Incorrect assertions
   (If the operation, formula, or approach itself does not hold — regardless of whether the arithmetic on top of it is executed correctly — that is a Method_ok problem, not a Calc_ok problem.)

Prior_ok) Consistency with this solution's own previous steps — Is the current step consistent with:
   • The previous steps in this solution
   • Any constraints or conditions established earlier in this solution
   Score 0 only when the current step actually contradicts something this solution has already established. Merely continuing from an earlier mistake, without contradicting it, is not a Prior_ok problem — the earlier mistake is scored where it occurred, not here. If there are no previous steps yet, this criterion is automatically satisfied (score 1).

Domain_ok) Consistency with problem/domain — Is the current step consistent with:
   • The established ground truth of the problem
   • The mathematical domain applicable to this problem
   • The conditions explicitly given in the problem statement
   A Method_ok=0 step is not automatically a Domain_ok=0 step: choosing the wrong method is not, by itself, a Domain_ok problem. Only score Domain_ok=0 when there is a clear, explicit conflict with the problem's stated conditions, established facts, or a theorem's applicability conditions.

Double check all four criteria explicitly in your reasoning. In your analysis, be sensitive to subtle issues like missing pre-requisites/assumptions, correct-looking statements with slight errors and high confidence statements containing errors.

IMPORTANT POINTS:

- If you find ANY error, even a minor one, you MUST assign a score of 0 to the appropriate criterion. Be skeptical and verify all claims thoroughly.

- By default, assign 0 to only the single most relevant category. Assign 0 to more than one category only when there are genuinely independent errors that are each, on their own, sufficient to justify a 0 — not merely because one error's description touches on more than one criterion's wording.

Reminder before you answer: by default, only the single most relevant category gets a 0. Use more than one 0 only for genuinely independent, each-sufficient-on-its-own errors.

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
