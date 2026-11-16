# Choosing the language models: evals

Which Groq model should each agent use? This write-up answers that with measurements on the
app's own tasks: what was measured, how the scoring was kept honest, every model that was
tried, the results, and the limits of all of it. To run the evals yourself, see
[backend/evals/README.md](../backend/evals/README.md).

## Models tried

| Model | Outcome |
| :--- | :--- |
| `openai/gpt-oss-20b`, low and medium reasoning | Full runs. **Chosen for resume writing** (low). |
| `openai/gpt-oss-120b`, low and medium reasoning | Full runs. **Chosen for every other agent** (medium). |
| `qwen/qwen3.8-27b` | One partial run, then dropped: it broke tool-call schemas, and Groq's free tier could not run it (see [Qwen 3.8 27B](#qwen-38-27b)). |
| `llama-3.3-70b-versatile` | The app's original default. Groq retired it before the evals began: every call failed with 404 `model_not_found`, so it could not be measured. |
| `gemini-3.6-flash` | Not a candidate: the judge for open-ended outputs (see [The judge](#the-judge)). |

Those were all the Groq chat models available that fit the agents; the rest were guard,
speech and transcription models. Prices per million tokens, input / output: 20b $0.075 / $0.30,
120b $0.15 / $0.60, Qwen about $0.80 / $4.00 (from third-party trackers; Groq's page did not
list it).

## What is measured

Twelve tasks, one per place an agent calls a model. Every case uses the agent's real prompt
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
One full run of every task uses about 55,000 tokens and costs under one cent on any setup.

## Why the numbers can be trusted

- **Floors.** The best score a constant answer gets ("always moderate", "always approve",
  one fixed tag set) is measured too. No constant answer passes more than half of any task,
  and a model's score is read against its floor.
- **An independent, checked judge** (below).
- **Repeats.** Each setup runs 3 times, and the summary shows the range of per-run means, so
  a difference smaller than the run-to-run spread is not read as a win.
- **Failures are not hidden.** The agents catch provider errors and fall back (a default
  grade, "Let's move on", a keyword matcher). The harness counts every failed call, so a
  fallback scores 0 instead of passing quietly. Replies the app would misread (no grade, an
  echoed "[bad/moderate/good/excellent]", no clear decision) fail the case.
- **Quota is not the model's fault.** A rate-limited (HTTP 429) case is left unscored and
  counted separately, unless a real failure happened in it too.
- **Only the model's own decisions count.** The interview moves on by itself after a
  non-answer such as "I don't know". Those cases are run but left unscored ("decided by the
  app"), so a dead model cannot pass them, and the floor counts only the cases the model decides.
- **Reproducible.** Every results file records the commit, models, prices, judge
  model and a fingerprint of each labelled-data file.
- **The harness is tested.** 160 offline tests, written alongside repeated rounds of
  adversarial review that probed it with fake models and code mutations. The review found real
  holes, for example a matcher that let one long item claim every skill, and a dead model
  passing 3 of 7 decision cases through the app's fallbacks.

## The judge

Open-ended outputs (follow-up questions, summaries, resumes, reviews) are scored by Gemini,
a different model family from every candidate, blind to which model wrote them.

- **Calibrated first.** Before its scores counted, it had to tell deliberately bad outputs
  from good ones: a praise-only, off-topic follow-up; a summary inventing topics; a resume
  inventing a Google job and a "300%" result; a review claiming skills the resume lacks. It
  scored the good outputs 0.92 to 1.00 and the bad ones 0.00 to 0.06.
- **Strict pass rule.** A judged case passes only if the mean is at least 0.75 *and* no
  criterion scores below 3 of 5, so a well-written resume with one invented achievement fails.
  A verdict that skips, adds or repeats criteria is rejected.
- **Why Flash.** The free tier of Google AI Studio gives no quota for the Pro models
  (`gemini-2.5-pro` is no longer offered to new users), and `gemini-3.6-flash` was the only
  Flash model answering at the time. Its limit of 20 requests a day per model is why only the
  first run of each setup was judged.

## Results

Four setups, 3 runs each, first run judged. 20b at medium has 2 runs: its third hit Groq's
daily token limit. Cells are passed/scored cases summed over the runs; the floor is the best
a constant answer does. The follow_up_decision row was measured on the first 8 cases, before
the app decided non-answers by itself (see the re-run below).

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

**Follow-up decisions, re-run.** On the 16 cases (12 decided by the model, floor 6/12, 3
runs), 120b at medium scored 30/36 and 20b at low 18/36, the floor. 20b asked a follow-up
after every answer, even plainly wrong ones. 120b moved on from every bad answer, including
the two that were on the right track and deserved a follow-up, so it routes by the grade
rather than the answer.

### Chosen models

| Setting | Model | Why |
| :--- | :--- | :--- |
| `LLM_MODEL` (every agent except resume writing) | `openai/gpt-oss-120b`, medium effort | Best or tied best on skills (14/15), grading (0.88 vs 0.72 for 20b), follow-up decisions (21/24), course parsing (6/6), the planner and the end-to-end path |
| `RESUME_LLM_MODEL` (resume draft and LaTeX) | `openai/gpt-oss-20b`, low effort | Ties 120b low on resumes at half the price and compiled both; 120b medium skipped the required tool call on 5 of 6 drafts |

A separate model per agent would gain little (critic 21/24 instead of 19/24 with 20b low,
tagging 42/45 instead of 39/45 with 120b low) for five settings instead of two.

### Qwen 3.8 27B

Qwen ran once, in an early single run of the seven tasks that need no judge, next to
the four gpt-oss setups. Scores are 0 to 1, from one run each:

| Task | 20b low | 20b medium | 120b low | 120b medium | Qwen |
| :--- | :--- | :--- | :--- | :--- | :--- |
| skill_extraction | 0.97 | 1.00 | 0.99 | 0.99 | 0.98 |
| template_tagging | 0.90 (7/10) | 0.90 (7/10) | 0.90 (7/10) | 0.87 (6/10) | 0.97 (9/10) |
| answer_grading | 0.72 (7/16) | 0.75 (8/16) | 0.91 (13/16) | 0.88 (12/16) | 0.78 (9/16) |
| follow_up_decision | 0.86 (6/7) | 0.57 (4/7) | 0.86 (6/7) | 1.00 (7/7) | 1.00 (7/7) |
| curriculum_planner | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| course_parsing | 0.83 (1/2) | 0.83 (1/2) | 1.00 | 1.00 | 0.00 (2 errors) |
| curriculum_critic | 1.00 (5/5) | 0.80 (4/5) | 0.80 (4/5) | 0.80 (4/5) | 0.80 (1 error) |

These used earlier, smaller case sets, so they are not comparable with the table above.
Qwen tagged templates best and decided follow-ups well, but it kept breaking tool-call
schemas: `is_approved` sent as a string, `path` missing from `submit_learning_path`, calls to
tools that do not exist (`Search`, `NewcourseList` for `NewCourseList`), and empty course
extraction. The later 3-run attempt could not be scored at all: Groq's free tier caps
Qwen at 1,000 output tokens a minute, single agent requests asked for more, and most calls
were refused before the model ran. At 5 to 10 times the price of gpt-oss, it was dropped.

## What the evals found

- **Higher effort is not always better.** 20b medium broke structured output on most resume
  and review calls; 120b medium replied in plain text instead of calling the resume tool.
- **gpt-oss-20b's grading is skewed:** it marks moderate answers "bad" and good ones
  "excellent". 120b's misses are one grade off.
- **Shared failures point at prompts, not models.** Every model asked a follow-up after "I
  don't know" or "just tell me the answer", where moving on is right. The app now moves on
  from non-answers by itself.
- **Open issues the evals surfaced:** 120b routes follow-ups by grade rather than by the
  answer; resume summaries can embellish ("Expert in"); the curriculum critic's prompt
  contradicts itself in places; the curriculum agent can invent course URLs when the catalog
  has none.

## Limits

- **Small case sets** (2 to 16 per task): results show clear differences, not fine ones.
- **One judge model**, Gemini 3.6 Flash, on its free tier. Only the first run of each setup
  was judged.
- **Labels written for this app**, not taken from a public benchmark, so other people's data
  may behave differently.
- **Free-tier quotas set the size of the study.** Groq's free tier allows 1,000 requests and
  200,000 tokens a day per model (a rolling 24 hours), about three full runs. One 120b run had
  to be thrown away and redone after hitting that limit midway, and 20b at medium got only two
  runs.
- **A snapshot in time.** Groq retires models (as it did Llama 3.3 70B) and their behaviour
  can change; the results files record the commit of every run.
