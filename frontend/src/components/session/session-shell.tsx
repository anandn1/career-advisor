"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { useParams, usePathname } from "next/navigation";
import { createContext, useContext, useEffect, type ReactNode } from "react";

import { RouteBar, type StationStatus } from "@/components/transit/route-bar";
import { forgetSession, keys, type Report, rememberSession, useReport, useSession } from "@/hooks/use-career";
import { ApiError, api, type Run, type Session, unwrap } from "@/lib/api/client";
import { STATIONS, type StationId } from "@/lib/stations";

const SessionContext = createContext<Session | null>(null);

export function useCurrentSession(): Session {
  const session = useContext(SessionContext);
  if (!session) throw new Error("useCurrentSession must be used inside SessionShell");
  return session;
}

function stationStatus(report: Report | undefined, session: Session, runs: Run[] = []): Partial<Record<StationId, StationStatus>> {
  const status: Partial<Record<StationId, StationStatus>> = {
    skills: session.skills.length > 0 ? "done" : "open",
    market: report?.job_market ? "done" : "open",
    path: report?.learning_path ? "done" : "open",
    resume: report?.resume || report?.resume_review ? "done" : "open",
    interview: report?.interviews.some((i) => i.status === "succeeded") ? "done" : "open",
  };
  for (const run of runs) {
    if (run.status !== "queued" && run.status !== "running") continue;
    const owner = STATIONS.find((station) => station.kinds.includes(run.kind));
    if (owner && run.kind !== "interview") status[owner.id] = "running";
  }
  return status;
}

function currentStation(pathname: string): StationId | "report" {
  const last = pathname.split("/").filter(Boolean).at(-1);
  return STATIONS.some((station) => station.id === last) ? (last as StationId) : "report";
}

function Notice({ title, children }: { title: string; children: ReactNode }) {
  return (
    <main className="mx-auto flex max-w-xl flex-1 flex-col justify-center gap-4 px-6 py-16">
      <h1 className="font-heading text-3xl font-extrabold">{title}</h1>
      <div className="text-lg text-ink-muted">{children}</div>
    </main>
  );
}

export function SessionShell({ children }: { children: ReactNode }) {
  const { sessionId } = useParams<{ sessionId: string }>();
  const pathname = usePathname();
  const session = useSession(sessionId);
  const report = useReport(sessionId);
  const runs = useQuery({
    queryKey: keys.runs(sessionId),
    queryFn: async () =>
      unwrap(await api.GET("/api/v1/sessions/{session_id}/runs", { params: { path: { session_id: sessionId } } })),
    refetchInterval: (query) =>
      query.state.data?.some((run) => run.status === "queued" || run.status === "running") ? 3_000 : false,
  });

  const missing = session.error instanceof ApiError && session.error.status === 404;
  useEffect(() => {
    if (session.data) rememberSession(sessionId);
    if (missing) forgetSession();
  }, [session.data, missing, sessionId]);

  if (session.isPending) return <p className="p-8 text-ink-muted">Loading your route…</p>;
  if (session.error) {
    return (
      <Notice title={missing ? "This route has expired" : "Can't reach the server"}>
        {missing ? (
          <p>Routes are deleted after 30 days without a visit, or when you delete them. <Link className="underline" href="/">Start a new route</Link>.</p>
        ) : (
          <p>{session.error.message} Check that the API is running, then reload.</p>
        )}
      </Notice>
    );
  }
  return (
    <SessionContext.Provider value={session.data}>
      <header className="sticky top-0 z-20 border-b-2 border-ink bg-paper/95 backdrop-blur">
        <div className="mx-auto flex w-full max-w-6xl flex-col gap-1 px-4 sm:px-6">
          <div className="flex flex-wrap items-center justify-between gap-2 pt-3 text-sm">
            <Link href="/" className="signage text-xs text-ink-muted hover:text-ink">Agentic AI Career Advisor</Link>
            <span className="text-ink-muted">
              {session.data.target_role} · {session.data.location}
            </span>
          </div>
          <RouteBar
            sessionId={sessionId}
            current={currentStation(pathname)}
            status={stationStatus(report.data, session.data, runs.data)}
            terminus={session.data.target_role}
          />
        </div>
      </header>
      <main className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-8 px-4 py-8 sm:px-6">{children}</main>
    </SessionContext.Provider>
  );
}
