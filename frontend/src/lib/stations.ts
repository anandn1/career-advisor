import type { Run, RunKind } from "@/lib/api/client";
import agentSteps from "@/lib/agent-steps.json";

/** A station on the route from the user's skills to the target role. */
export type StationId = "skills" | "market" | "path" | "resume" | "interview";

export type Station = {
  id: StationId;
  name: string;
  letter: string;
  /** CSS colour token for the station's line. */
  colour: string;
  /** Text colour that stays readable on the line colour. */
  onColour: string;
  kinds: RunKind[];
};

export const STATIONS: Station[] = [
  { id: "skills", name: "Skills", letter: "S", colour: "var(--color-line-skills)", onColour: "#fff",
    kinds: ["skills_from_resume", "skills_from_github"] },
  { id: "market", name: "Market", letter: "M", colour: "var(--color-line-market)", onColour: "#fff",
    kinds: ["job_market"] },
  { id: "path", name: "Learning path", letter: "P", colour: "var(--color-line-path)", onColour: "#fff",
    kinds: ["learning_path"] },
  { id: "resume", name: "Resume", letter: "R", colour: "var(--color-line-resume)", onColour: "#fff",
    kinds: ["resume", "resume_render", "resume_review"] },
  { id: "interview", name: "Interview", letter: "I", colour: "var(--color-line-interview)",
    onColour: "var(--color-ink)", kinds: ["interview"] },
];

export function station(id: StationId): Station {
  const found = STATIONS.find((candidate) => candidate.id === id);
  if (!found) throw new Error(`Unknown station ${id}`);
  return found;
}

/** A stop on a run line: one readable step, made of one or more agent graph nodes. */
export type Stop = { id: string; label: string; nodes: string[] };

type BackgroundKind = Exclude<RunKind, "interview">;

/**
 * Agent step names per stop. They live in agent-steps.json because the backend test suite
 * checks them against the real LangGraph nodes and handler progress steps.
 */
const STEP_NAMES = agentSteps as Record<BackgroundKind, Record<string, string[]>>;

const STOP_LABELS: Record<BackgroundKind, Record<string, string>> = {
  skills_from_resume: { read: "Reading your resume" },
  skills_from_github: { read: "Reading your GitHub" },
  job_market: { cache: "Saved postings", scrape: "Live LinkedIn postings", gaps: "Skill gaps" },
  learning_path: { plan: "Planning", search: "Finding courses", score: "Scoring courses",
                   assemble: "Assembling the path", review: "Reviewing" },
  resume: { draft: "Writing your resume", pdf: "Typesetting the PDF" },
  resume_render: { typeset: "Typesetting your edits", pdf: "Typesetting the PDF" },
  resume_review: { review: "Reviewing your resume" },
};

export const STOPS = Object.fromEntries(
  (Object.keys(STOP_LABELS) as BackgroundKind[]).map((kind) => [
    kind,
    Object.entries(STOP_LABELS[kind]).map(([id, label]) => ({ id, label, nodes: STEP_NAMES[kind][id] })),
  ]),
) as Record<BackgroundKind, Stop[]>;

export type StopState = "waiting" | "current" | "passed";

/** `visits`: how many times the run entered this stop (the learning path loops once per skill). */
export type StopProgress = { stop: Stop; state: StopState; visits: number };

/** Where a run is on its line, derived only from its recorded progress events. */
export function stopProgress(run: Pick<Run, "kind" | "status" | "events">): StopProgress[] {
  if (run.kind === "interview") return [];
  const stops = STOPS[run.kind];
  const visits = new Map<string, number>();
  let furthest = -1; // the furthest stop reached so far
  let latest = -1; // the stop of the most recent step (the learning path loops back)
  for (const event of run.events) {
    const index = stops.findIndex((stop) => stop.nodes.includes(String(event.step)));
    if (index === -1) continue;
    if (index !== latest) visits.set(stops[index].id, (visits.get(stops[index].id) ?? 0) + 1);
    latest = index;
    furthest = Math.max(furthest, index);
  }
  const travelling = run.status === "queued" || run.status === "running";
  return stops.map((stop, index) => {
    let state: StopState = "waiting";
    if (travelling && index === latest) state = "current";
    else if (index <= furthest && (index < furthest || !travelling || index !== latest)) state = "passed";
    return { stop, state, visits: visits.get(stop.id) ?? 0 };
  });
}

export function isFinished(run: Pick<Run, "status">): boolean {
  return run.status === "succeeded" || run.status === "failed";
}
