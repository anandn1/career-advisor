import type { Run, RunKind } from "@/lib/api/client";

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

export const STOPS: Record<Exclude<RunKind, "interview">, Stop[]> = {
  skills_from_resume: [{ id: "read", label: "Reading your resume", nodes: ["extracting_skills"] }],
  skills_from_github: [{ id: "read", label: "Reading your GitHub", nodes: ["reading_github"] }],
  job_market: [
    { id: "cache", label: "Saved postings", nodes: ["search_cached_postings"] },
    { id: "scrape", label: "Live LinkedIn postings", nodes: ["scrape_fresh_postings"] },
    { id: "gaps", label: "Skill gaps", nodes: ["analyze_skill_gaps"] },
  ],
  learning_path: [
    { id: "plan", label: "Planning", nodes: ["entry", "planner"] },
    { id: "search", label: "Finding courses",
      nodes: ["iteration_controller", "fetch_course_catalog", "web_search_for_courses",
              "process_and_store_courses"] },
    { id: "score", label: "Scoring courses",
      nodes: ["start_evaluation", "check_prerequisites", "match_interests", "estimate_workload",
              "aggregate_scores", "recommend_enrollment", "suggest_alternatives"] },
    { id: "assemble", label: "Assembling the path", nodes: ["architect", "handle_submission"] },
    { id: "review", label: "Reviewing", nodes: ["critic", "prepare_revision"] },
  ],
  resume: [
    { id: "draft", label: "Writing your resume", nodes: ["drafting_resume"] },
    { id: "pdf", label: "Typesetting the PDF", nodes: ["compiling_pdf"] },
  ],
  resume_render: [
    { id: "typeset", label: "Typesetting your edits", nodes: ["typesetting_resume"] },
    { id: "pdf", label: "Typesetting the PDF", nodes: ["compiling_pdf"] },
  ],
  resume_review: [{ id: "review", label: "Reviewing your resume", nodes: ["reviewing_resume"] }],
};

export type StopState = "waiting" | "current" | "passed";

/** `visits`: how many times the run entered this stop (the learning path loops once per skill). */
export type StopProgress = { stop: Stop; state: StopState; visits: number };

/** Where a run is on its line, derived only from its recorded progress events. */
export function stopProgress(run: Pick<Run, "kind" | "status" | "events">): StopProgress[] {
  if (run.kind === "interview") return [];
  const stops = STOPS[run.kind];
  const visits = new Map<string, number>();
  let lastStop = -1;
  let previous = -1;
  for (const event of run.events) {
    const index = stops.findIndex((stop) => stop.nodes.includes(String(event.step)));
    if (index === -1) continue;
    if (index !== previous) visits.set(stops[index].id, (visits.get(stops[index].id) ?? 0) + 1);
    previous = index;
    lastStop = Math.max(lastStop, index);
  }
  const finished = run.status === "succeeded" || run.status === "failed";
  return stops.map((stop, index) => {
    let state: StopState = "waiting";
    if (index < lastStop || (finished && visits.has(stop.id))) state = "passed";
    else if (index === lastStop) state = finished ? "passed" : "current";
    return { stop, state, visits: visits.get(stop.id) ?? 0 };
  });
}

export function isFinished(run: Pick<Run, "status">): boolean {
  return run.status === "succeeded" || run.status === "failed";
}
