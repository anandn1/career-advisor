"use client";

import Link from "next/link";

import { RunPanel } from "@/components/session/run-panel";
import { useCurrentSession } from "@/components/session/session-shell";
import { StationHeader } from "@/components/session/station-header";
import { useLatestRun } from "@/components/stations/use-latest-run";
import { Button } from "@/components/ui/button";
import { type LearningPath, useReport, useStartRun } from "@/hooks/use-career";
import { api, unwrap } from "@/lib/api/client";
import { isFinished, station } from "@/lib/stations";

const PATH = station("path");

export function groupBySkill(path: LearningPath) {
  const groups = new Map<string, LearningPath["recommendations"]>();
  for (const course of path.recommendations) {
    groups.set(course.skill_name, [...(groups.get(course.skill_name) ?? []), course]);
  }
  return [...groups.entries()];
}

function PathResults({ path }: { path: LearningPath }) {
  return (
    <div className="flex flex-col gap-6">
      <p className="max-w-3xl text-lg">{path.user_summary}</p>
      <ol className="flex flex-col gap-6">
        {groupBySkill(path).map(([skill, courses]) => (
          <li key={skill} className="flex flex-col gap-2 border-l-[5px] pl-4" style={{ borderColor: PATH.colour }}>
            <h2 className="font-heading text-xl font-extrabold">{skill}</h2>
            <ul className="flex flex-col gap-2">
              {courses.map((course) => (
                <li key={course.course_url + course.course_title} className="rounded-lg bg-surface p-3">
                  <a href={course.course_url} target="_blank" rel="noreferrer"
                     className="font-bold text-line-path underline-offset-2 hover:underline">
                    {course.course_title}
                  </a>
                  <p className="text-sm text-ink-muted">{course.description}</p>
                </li>
              ))}
            </ul>
          </li>
        ))}
      </ol>
    </div>
  );
}

export function PathStation() {
  const session = useCurrentSession();
  const report = useReport(session.id);
  const { latest } = useLatestRun(session.id, PATH.kinds);
  const start = useStartRun(session.id, async () =>
    unwrap(await api.POST("/api/v1/sessions/{session_id}/runs/learning-path", {
      params: { path: { session_id: session.id } },
    })));
  const gaps = report.data?.job_market?.analysis.skill_gaps ?? [];
  const busy = start.isPending || (latest !== undefined && !isFinished(latest));

  return (
    <>
      <StationHeader station={PATH} title="Courses for the skills you are missing">
        The agent plans which gaps matter most, searches a course catalog and the web, scores each course
        against your level, and a second agent reviews the path before you see it.
      </StationHeader>
      {gaps.length === 0 && !report.isPending ? (
        <p className="text-lg">
          This station needs your skill gaps first.{" "}
          <Link className="font-bold underline" href={`/s/${session.id}/market`}>Analyze the job market</Link>.
        </p>
      ) : (
        <div className="flex flex-wrap items-center gap-3">
          <Button size="lg" onClick={() => start.mutate(undefined)} disabled={busy || gaps.length === 0}
                  className="font-heading font-bold">
            {latest ? "Plan again" : "Plan my courses"}
          </Button>
          <span className="text-sm text-ink-muted">For {gaps.length} gaps: {gaps.slice(0, 6).join(", ")}{gaps.length > 6 ? "…" : ""}</span>
        </div>
      )}
      {start.error && <p role="alert" className="text-danger">{start.error.message}</p>}
      <RunPanel sessionId={session.id} station={PATH} runId={latest?.id}>
        {(run) => <PathResults path={(run.result as unknown as { path: LearningPath }).path} />}
      </RunPanel>
    </>
  );
}
