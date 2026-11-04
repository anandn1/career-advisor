"use client";

import type { ReactNode } from "react";

import { RunLine } from "@/components/transit/run-line";
import { useRun } from "@/hooks/use-career";
import type { Run } from "@/lib/api/client";
import { isFinished, type Station } from "@/lib/stations";

/** The live line for a run while it travels, then a short outcome; `children` shows its result. */
export function RunPanel({
  sessionId,
  station,
  runId,
  children,
}: {
  sessionId: string;
  station: Station;
  runId: string | undefined;
  children?: (run: Run) => ReactNode;
}) {
  const { data: run, error } = useRun(sessionId, runId);
  if (!run) {
    return error ? <p role="alert" className="text-danger">Couldn&apos;t load this run: {error.message}</p> : null;
  }
  const travelling = !isFinished(run);
  return (
    <section aria-live="polite" className="flex flex-col gap-4">
      {(travelling || run.status === "failed") && (
        <div className="rounded-xl border-2 border-rule bg-surface p-5">
          <p className="signage mb-3 text-xs text-ink-muted">{travelling ? "On the line now" : "Last run"}</p>
          <RunLine run={run} station={station} />
          {run.status === "failed" && (
            <p role="alert" className="mt-3 text-danger">{run.error ?? "The agent stopped. Try again."}</p>
          )}
        </div>
      )}
      {run.status === "succeeded" && children?.(run)}
    </section>
  );
}
