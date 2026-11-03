"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";

import {
  api,
  type Interview,
  type Run,
  type RunKind,
  type Session,
  runEventsUrl,
  unwrap,
} from "@/lib/api/client";
import { isFinished } from "@/lib/stations";

const SESSION_STORAGE_KEY = "career-advisor:session";

export const keys = {
  session: (id: string) => ["session", id] as const,
  report: (id: string) => ["report", id] as const,
  runs: (id: string, kind?: RunKind) => ["runs", id, kind ?? "all"] as const,
  run: (id: string, runId: string) => ["run", id, runId] as const,
  interview: (id: string, interviewId: string) => ["interview", id, interviewId] as const,
  templates: ["templates"] as const,
};

export function rememberSession(id: string) {
  window.localStorage.setItem(SESSION_STORAGE_KEY, id);
}

export function rememberedSession(): string | null {
  return typeof window === "undefined" ? null : window.localStorage.getItem(SESSION_STORAGE_KEY);
}

export function forgetSession() {
  window.localStorage.removeItem(SESSION_STORAGE_KEY);
}

const path = (sessionId: string) => ({ params: { path: { session_id: sessionId } } });

export function useSession(sessionId: string) {
  return useQuery({
    queryKey: keys.session(sessionId),
    queryFn: async () => unwrap(await api.GET("/api/v1/sessions/{session_id}", path(sessionId))),
  });
}

export function useReport(sessionId: string) {
  return useQuery({
    queryKey: keys.report(sessionId),
    queryFn: async () =>
      unwrap(await api.GET("/api/v1/sessions/{session_id}/report", path(sessionId))) as Report,
  });
}

/** The fields of /report the UI reads; each section holds the latest successful run's result. */
export type Report = {
  session: Session;
  job_market: ({ run_id: string } & JobMarketResult) | null;
  learning_path: ({ run_id: string } & { path: LearningPath }) | null;
  resume: ({ run_id: string } & ResumeResult) | null;
  resume_review: ({ run_id: string } & { review: ResumeReview }) | null;
  interviews: { id: string; status: string; summary: string | null }[];
};

export type Job = {
  title: string;
  company: string;
  location: string;
  skills_required: string[];
  source?: "live" | "cached";
  scraped_at?: string | null;
};
export type JobMarketResult = {
  jobs: Job[];
  analysis: {
    skill_gaps?: string[];
    all_required_skills?: string[];
    average_gaps_per_job?: number;
    gaps_per_job?: { title: string; company: string; gaps: string[]; gap_count: number }[];
  };
};
export type LearningPath = {
  user_summary: string;
  skill_gaps: string[];
  recommendations: { skill_name: string; course_title: string; course_url: string; description: string }[];
};
export type ResumeResult = {
  draft: import("@/lib/api/client").ResumeDraft;
  template_id: string | null;
  has_pdf: boolean;
  pdf_error: string | null;
};
export type ResumeReview = {
  summary: string;
  strengths: string[];
  weaknesses: string[];
  suggestions: string[];
};

export function useRuns(sessionId: string, kinds: RunKind[]) {
  return useQuery({
    queryKey: [...keys.runs(sessionId), ...kinds],
    queryFn: async () => {
      const runs = unwrap(await api.GET("/api/v1/sessions/{session_id}/runs", path(sessionId)));
      return runs.filter((run) => kinds.includes(run.kind));
    },
  });
}

/** Refresh everything a finished run can change. */
function invalidateAfterRun(queryClient: ReturnType<typeof useQueryClient>, sessionId: string) {
  for (const key of [keys.session(sessionId), keys.report(sessionId), ["runs", sessionId]]) {
    queryClient.invalidateQueries({ queryKey: key });
  }
}

/** Put `event` at position `index`, replacing what is there; append if the index is unusable. */
export function placeEvent<T>(events: T[], index: number, event: T): T[] {
  if (!Number.isInteger(index) || index < 0 || index >= events.length) return [...events, event];
  return events.map((existing, position) => (position === index ? event : existing));
}

/** Live progress for an unfinished run: SSE events patch the cached run until "done". */
export function useRunStream(sessionId: string, run: Run | undefined) {
  const queryClient = useQueryClient();
  const runId = run?.id;
  const finished = run ? isFinished(run) : true;

  useEffect(() => {
    if (!runId || finished) return;
    const source = new EventSource(runEventsUrl(sessionId, runId));
    const patch = (update: (current: Run) => Run) =>
      queryClient.setQueryData<Run>(keys.run(sessionId, runId), (current) =>
        current ? update(current) : current,
      );
    source.addEventListener("progress", (message) => {
      const { data, lastEventId } = message as MessageEvent;
      // The id is the event's position. The first connection sends every event; a reconnect
      // resumes after Last-Event-ID. An event without an id is appended.
      const index = lastEventId === "" ? Number.NaN : Number(lastEventId);
      patch((current) => ({
        ...current,
        status: "running",
        events: placeEvent(current.events, index, JSON.parse(data)),
      }));
    });
    source.addEventListener("done", (message) => {
      queryClient.setQueryData<Run>(keys.run(sessionId, runId), JSON.parse((message as MessageEvent).data));
      source.close(); // otherwise the browser reconnects to a finished stream
      invalidateAfterRun(queryClient, sessionId);
    });
    return () => source.close();
  }, [sessionId, runId, finished, queryClient]);
}

export function useRun(sessionId: string, runId: string | undefined) {
  const query = useQuery({
    queryKey: keys.run(sessionId, runId ?? "none"),
    queryFn: async () =>
      unwrap(
        await api.GET("/api/v1/sessions/{session_id}/runs/{run_id}", {
          params: { path: { session_id: sessionId, run_id: runId! } },
        }),
      ),
    enabled: Boolean(runId),
  });
  useRunStream(sessionId, query.data);
  return query;
}

/** Start a run with a request already bound to its endpoint; seeds the run cache. */
export function useStartRun(sessionId: string, start: (body?: unknown) => Promise<Run>) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: start,
    onSuccess: (run) => {
      queryClient.setQueryData(keys.run(sessionId, run.id), run);
      queryClient.invalidateQueries({ queryKey: ["runs", sessionId] });
      if (isFinished(run)) invalidateAfterRun(queryClient, sessionId);
    },
  });
}

export function useUpdateSession(sessionId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (body: { target_role?: string; location?: string; skills?: string[] }) =>
      unwrap(await api.PATCH("/api/v1/sessions/{session_id}", { ...path(sessionId), body })),
    onSuccess: (session) => {
      queryClient.setQueryData(keys.session(sessionId), session);
      queryClient.invalidateQueries({ queryKey: keys.report(sessionId) });
    },
  });
}

export function useTemplates() {
  return useQuery({
    queryKey: keys.templates,
    queryFn: async () => unwrap(await api.GET("/api/v1/templates")),
  });
}

export function useInterview(sessionId: string, interviewId: string | undefined) {
  return useQuery<Interview>({
    queryKey: keys.interview(sessionId, interviewId ?? "none"),
    queryFn: async () =>
      unwrap(
        await api.GET("/api/v1/sessions/{session_id}/interviews/{interview_id}", {
          params: { path: { session_id: sessionId, interview_id: interviewId! } },
        }),
      ),
    enabled: Boolean(interviewId),
  });
}
