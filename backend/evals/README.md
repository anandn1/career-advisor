# Model evals

Which Groq model should each agent use? These evals answer that with measurements on the
app's own tasks, instead of guesses. They run each candidate model through the exact code
path an agent uses, score the result, and record what it cost.

**The results, the models tried, how the scoring works and its limits are in
[docs/EVALS.md](../../docs/EVALS.md).** This page is about running the harness.

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

## Layout

- `tasks/`: the tasks, grouped by agent: how a case runs through the agent's real code and
  how it is scored.
- `data/`: the hand-checked cases and labels as JSON, plus the judge's calibration pairs.
- `core.py`: what every task shares: loading cases, running one, recording the result.
- `models.py`: the candidate Groq models, their prices, and swapping them into the agents.
- `judge.py`, `calibration.py`: the Gemini judge and its calibration check.
- `matching.py`: matching returned skills and tags against the labels.
- `report.py`: results files and the Markdown summary. `baselines.py`: the constant-answer floors.
