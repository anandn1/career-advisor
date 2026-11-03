"use client";

import { RunPanel } from "@/components/session/run-panel";
import { useCurrentSession } from "@/components/session/session-shell";
import { StationHeader } from "@/components/session/station-header";
import { useLatestRun } from "@/components/stations/use-latest-run";
import { Button } from "@/components/ui/button";
import { type Job, type JobMarketResult, useStartRun } from "@/hooks/use-career";
import { api, unwrap } from "@/lib/api/client";
import { isFinished, station } from "@/lib/stations";

const MARKET = station("market");

/** Gaps ranked by how many postings ask for them. */
export function rankGaps(result: JobMarketResult): { skill: string; postings: number }[] {
  const counts = new Map<string, number>();
  for (const job of result.analysis.gaps_per_job ?? []) {
    for (const gap of job.gaps) counts.set(gap, (counts.get(gap) ?? 0) + 1);
  }
  for (const gap of result.analysis.skill_gaps ?? []) if (!counts.has(gap)) counts.set(gap, 0);
  return [...counts.entries()]
    .map(([skill, postings]) => ({ skill, postings }))
    .sort((a, b) => b.postings - a.postings || a.skill.localeCompare(b.skill));
}

function sourceLabel(job: Job): string {
  if (job.source === "cached") return job.scraped_at ? `Saved ${job.scraped_at}` : "Saved earlier";
  return "Live";
}

function MarketResults({ result }: { result: JobMarketResult }) {
  const gaps = rankGaps(result);
  const total = result.jobs.length;
  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
      <section className="flex flex-col gap-3">
        <h2 className="signage text-sm">Skills you are missing</h2>
        {gaps.length === 0 ? (
          <p className="text-ink-muted">You already cover every skill these postings ask for.</p>
        ) : (
          <ol className="grid grid-cols-[minmax(5rem,auto)_1fr_auto] items-center gap-x-3 gap-y-2">
            {gaps.map(({ skill, postings }) => (
              <li key={skill} className="col-span-3 grid grid-cols-subgrid items-center">
                <span className="truncate font-heading font-bold">{skill}</span>
                <span aria-hidden className="h-2.5 rounded-full bg-rule/60">
                  <span className="block h-full rounded-full" style={{
                    background: MARKET.colour,
                    width: `${total ? Math.max(4, (postings / total) * 100) : 4}%`,
                  }} />
                </span>
                <span className="font-mono text-xs whitespace-nowrap text-ink-muted">{postings} of {total}</span>
              </li>
            ))}
          </ol>
        )}
      </section>
      <section className="flex flex-col gap-3">
        <h2 className="signage text-sm">Postings read ({total})</h2>
        {total === 0 ? (
          <p className="text-ink-muted">No postings were found for this role and place. Try a broader location such as a country or Remote.</p>
        ) : (
          <ul className="flex flex-col divide-y-2 divide-rule rounded-xl border-2 border-rule bg-surface">
            {result.jobs.map((job, index) => (
              <li key={`${job.company}-${job.title}-${index}`} className="flex flex-col gap-1.5 p-4">
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <span className="font-heading font-bold">{job.title}</span>
                  <span className="signage text-[0.65rem] text-ink-muted">{sourceLabel(job)}</span>
                </div>
                <span className="text-sm text-ink-muted">{job.company} · {job.location}</span>
                {job.skills_required.length > 0 && (
                  <p className="font-mono text-xs text-ink-muted">{job.skills_required.join(" · ")}</p>
                )}
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

export function MarketStation() {
  const session = useCurrentSession();
  const { latest } = useLatestRun(session.id, MARKET.kinds);
  const start = useStartRun(session.id, async () =>
    unwrap(await api.POST("/api/v1/sessions/{session_id}/runs/job-market", {
      params: { path: { session_id: session.id } },
    })));
  const busy = start.isPending || (latest !== undefined && !isFinished(latest));

  return (
    <>
      <StationHeader station={MARKET} title={`Who is hiring ${session.target_role}s in ${session.location}`}>
        The agent reads saved and live LinkedIn postings, pulls out the skills they ask for and compares them with yours.
      </StationHeader>
      <div className="flex flex-wrap items-center gap-3">
        <Button size="lg" onClick={() => start.mutate(undefined)} disabled={busy} className="font-heading font-bold">
          {latest ? "Analyze again" : "Analyze the job market"}
        </Button>
        {session.skills.length === 0 && (
          <span className="text-sm text-ink-muted">Tip: add your skills first, or every requirement counts as a gap.</span>
        )}
      </div>
      {start.error && <p role="alert" className="text-danger">{start.error.message}</p>}
      <RunPanel sessionId={session.id} station={MARKET} runId={latest?.id}>
        {(run) => <MarketResults result={run.result as unknown as JobMarketResult} />}
      </RunPanel>
    </>
  );
}
