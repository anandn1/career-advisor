# Model evals

Which Groq model should each agent use? These evals answer that with measurements on the
app's own tasks, instead of guesses. They run each candidate model through the exact code
path an agent uses, score the result, and record what it cost.

They are not part of `pytest`: they call Groq (and Gemini for judging), so they spend quota.
The harness itself is tested offline in `tests/test_evals.py`.

## Running them

From `backend/`, with `GROQ_API_KEY` and `GEMINI_API_KEY` in `.env`:

```bash
uv run python -m evals --models openai/gpt-oss-20b@low,openai/gpt-oss-120b@low --tasks all
uv run python -m evals --models openai/gpt-oss-20b@low --repeats 3 --defer-judging
uv run python -m evals --rejudge evals/results/<file>.json   # judge what was deferred
uv run python -m evals --rescore evals/results/<file>.json   # re-apply the pass rule, no calls
uv run python -m evals --calibrate-judge                    # check the judge first
uv run python -m evals --baselines                          # constant-answer floors
```

`model@effort` sets the reasoning effort of gpt-oss models (`low`, `medium`, `high`).
Resumes are compiled with local `pdflatex`, or inside the backend Docker image with
`--latex-image career-advisor-backend`. Results go to `evals/results/` (not committed).

## What is measured

Twelve tasks, one per LLM call site in the agents. Every case uses the agent's real prompt
and parser, with the candidate model swapped in.

| Agent | Task | Cases | How it is scored |
| :--- | :--- | ---: | :--- |
| Skills | `skill_extraction` | 5 | Recall and precision against hand-checked skill lists (accepted spellings, fair extras); pass = nothing missed, at most one extra |
| Templates | `template_tagging` | 15 | Required and forbidden tags per CV, including 5 non-tech CVs |
| Interview | `answer_grading` | 16 | Answers written at known quality; exact grade 1, one step off 0.5 |
| Interview | `follow_up_decision` | 16 | Cases with a clear right choice; the reply must name exactly one decision. The 4 non-answers ("I don't know", skip) are decided by the app itself and not scored |
| Interview | `follow_up_question` | 4 | Judge, plus one question and no praise opener |
| Interview | `interview_summary` | 2 | Judge: faithful to the report, covers each topic, useful |
| Curriculum | `curriculum_planner` | 3 | Every skill gap extracted, none invented, plausible level |
| Curriculum | `course_parsing` | 2 | Real courses kept; Reddit thread, blog post and job ad dropped; no invented URLs |
| Curriculum | `curriculum_critic` | 8 | Approve or reject, 4 of each |
| Curriculum | `curriculum_end_to_end` | 2 | Whole graph on catalog-covered gaps: every gap covered, catalog URLs only, critic approves |
| Resume | `resume_generation` | 2 | Facts kept, no invented numbers, no template text left in, compiles; judge reads the finished LaTeX |
| Resume | `resume_review` | 2 | Judge: grounded, role-specific, actionable, honest about gaps; 3-5 points per list |

Each case also records tokens (input, output, reasoning), cost at Groq's prices and time.

## Why the numbers can be trusted

- **Floors.** `--baselines` shows the best score a constant answer gets ("always
  moderate", "always approve", one fixed tag set). No constant answer passes more than half
  of any task; a model's score is read against its floor.
- **An independent, checked judge.** Open-ended outputs are scored by Gemini, a different
  model family from every candidate, blind to which model wrote them. Before its scores
  count, it must tell deliberately bad outputs (invented employers, praise-only follow-ups,
  summaries that invent topics) from good ones: it scored good 0.92-1.00 and bad 0.00-0.06.
  A verdict that skips, adds or repeats criteria is rejected. A judged case passes only if
  the mean is at least 0.75 *and* no criterion scores below 3 of 5, so a well-written resume
  with one invented achievement fails.
- **Repeats.** Each setup runs 3 times; the summary shows the range of per-run means, so a
  difference smaller than the run-to-run spread is not read as a win.
- **Failures are not hidden.** The agents catch provider errors and fall back (a default
  grade, "Let's move on", a keyword matcher). The harness counts every failed call, so a
  fallback scores 0 instead of passing quietly. Replies the app would misread (no grade, an
  echoed "[bad/moderate/good/excellent]", no clear decision) fail the case.
- **Quota is not the model's fault.** A rate-limited (HTTP 429) case is left unscored and
  counted separately, unless a real failure happened in it too.
- **Only the model's own decisions count.** The interview moves on by itself after a
  non-answer such as "I don't know". Those cases are run but left unscored ("decided by the
  app"), so a dead model cannot pass them; the floor counts only the cases the model decides.
- **Reproducible.** Every results file records the commit, date, models, prices, judge
  model and a fingerprint of each labelled-data file.
- **The harness is tested.** 160 offline tests, written alongside repeated rounds of
  adversarial review that probed it with fake models and code mutations. The review found real holes,
  for example a matcher that let one long item claim every skill, and a dead model passing
  3 of 7 decision cases through the app's fallbacks.

## Limits

- Small case sets (2 to 16 per task): results show clear differences, not fine ones.
- One judge model (Gemini 3.6 Flash; the free tier gives no Pro quota).
- The labels were written for this app, not taken from a public benchmark, so other
  people's data may behave differently.
- Groq's free tier allows about three full runs per model per day, which set the number of
  repeats.

## Results

Four setups, 3 runs each (20b@medium: 2, as its third run hit Groq's daily token limit),
first run judged. Cells show passed/scored cases; the floor is the best a constant answer
does. The follow_up_decision row was measured on the first 8 cases, before the app decided
non-answers by itself. Re-run on the 16 cases (12 decided by the model, floor 6/12, 3 runs):
120b at medium 30/36, 20b at low 18/36. 20b asked a follow-up after every answer, even plainly
wrong ones. 120b moved on from every bad answer, including the two that were on the right track
and deserved a follow-up, so it routes by the grade rather than the answer.

| Task | Floor | 20b low | 120b low | 20b medium | 120b medium |
| :--- | :--- | :--- | :--- | :--- | :--- |
| skill_extraction | | 9/15 | 9/15 | 7/10 | **14/15** |
| template_tagging | 7/15 | 39/45 | **42/45** | 25/30 | 39/45 |
| answer_grading (score) | 0.50 | 0.72 | **0.88** | 0.75 | **0.88** |
| follow_up_decision | 4/8 | 18/24 | 18/24 | 8/16 | **21/24** |
| follow_up_question | | 4/4 | 4/4 | 4/4 | 4/4 |
| interview_summary | | 2/2 | 2/2 | 1/1 | 2/2 |
| curriculum_planner | | 9/9 | 9/9 | 6/6 | 9/9 |
| course_parsing | | 3/6 | **6/6** | 1/4 | **6/6** |
| curriculum_critic | 4/8 | **21/24** | 18/24 | 15/16 | 19/24 |
| curriculum_end_to_end | | 4/6 | 4/6 | 2/4 | 4/6 |
| resume_generation | | **1/2** | **1/2** | 0/2 | 1/6 |
| resume_review | | 2/2 | 2/2 | 0/3 | 2/2 |

### Chosen models

| Setting | Model | Why |
| :--- | :--- | :--- |
| `LLM_MODEL` (every agent except resume writing) | `openai/gpt-oss-120b`, medium effort | Best or tied best on skills (14/15), grading (0.88 vs 0.72 for 20b), follow-up decisions (21/24), course parsing (6/6), the planner and the end-to-end path |
| `RESUME_LLM_MODEL` (resume draft and LaTeX) | `openai/gpt-oss-20b`, low effort | Ties 120b low on resumes at half the price and compiled both; 120b medium skipped the required tool call on 5 of 6 drafts |

A separate model per agent would gain little (critic 21/24 instead of 19/24 with 20b low,
tagging 42/45 instead of 39/45 with 120b low) for five settings instead of two. Cost does not
decide it: one full run of every task costs under $0.01 on any setup.

### What the evals found

- **Higher effort is not always better.** 20b medium broke structured output on most resume
  and review calls; 120b medium replied in plain text instead of calling the resume tool.
- **gpt-oss-20b's grading is skewed:** it marks moderate answers "bad" and good ones
  "excellent". 120b's misses are one grade off.
- **Shared failures point at prompts, not models.** Every model asks a follow-up after "I
  don't know" or "just tell me the answer", where moving on is right.
- **Retired and unusable models.** `llama-3.3-70b-versatile`, the old default, was retired
  by Groq. Qwen 3.8 27B broke tool-call schemas often and its free tier caps output below
  what the agents request.
