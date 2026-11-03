import { describe, expect, it } from "vitest";

import type { Run } from "@/lib/api/client";
import { isFinished, station, STATIONS, STOPS, stopProgress } from "@/lib/stations";

function run(kind: Run["kind"], status: Run["status"], steps: string[]): Pick<Run, "kind" | "status" | "events"> {
  return { kind, status, events: steps.map((step) => ({ step, detail: {}, at: "" })) };
}

const states = (r: ReturnType<typeof run>) => stopProgress(r).map((p) => p.state);

describe("stopProgress", () => {
  it("shows every stop waiting before any event", () => {
    expect(states(run("job_market", "queued", []))).toEqual(["waiting", "waiting", "waiting"]);
  });

  it("marks the latest stop current and earlier ones passed while running", () => {
    expect(states(run("job_market", "running", ["search_cached_postings", "scrape_fresh_postings"])))
      .toEqual(["passed", "current", "waiting"]);
  });

  it("marks a skipped earlier stop as passed", () => {
    expect(states(run("job_market", "running", ["scrape_fresh_postings"]))).toEqual(["passed", "current", "waiting"]);
  });

  it("marks visited stops passed once the run succeeds", () => {
    const all = ["search_cached_postings", "scrape_fresh_postings", "analyze_skill_gaps"];
    expect(states(run("job_market", "succeeded", all))).toEqual(["passed", "passed", "passed"]);
  });

  it("leaves unreached stops waiting when a run fails", () => {
    expect(states(run("job_market", "failed", ["search_cached_postings"]))).toEqual(["passed", "waiting", "waiting"]);
  });

  it("ignores steps that belong to no stop", () => {
    expect(states(run("job_market", "running", ["mystery_node"]))).toEqual(["waiting", "waiting", "waiting"]);
  });

  it("counts entries into a stop, not the events inside it", () => {
    const steps = ["entry", "planner", "iteration_controller", "fetch_course_catalog", "start_evaluation",
      "check_prerequisites", "iteration_controller", "fetch_course_catalog"];
    const visits = Object.fromEntries(stopProgress(run("learning_path", "running", steps)).map((p) => [p.stop.id, p.visits]));
    expect(visits).toEqual({ plan: 1, search: 2, score: 1, assemble: 0, review: 0 });
  });

  it("moves the current stop back when the learning path loops to the next skill", () => {
    const steps = ["planner", "iteration_controller", "fetch_course_catalog", "start_evaluation", "check_prerequisites",
      "iteration_controller"];
    expect(states(run("learning_path", "running", steps))).toEqual(["passed", "current", "passed", "waiting", "waiting"]);
  });

  it("marks every reached stop passed once a looping run finishes", () => {
    const steps = ["planner", "iteration_controller", "start_evaluation", "iteration_controller"];
    expect(states(run("learning_path", "succeeded", steps))).toEqual(["passed", "passed", "passed", "waiting", "waiting"]);
  });

  it("returns no stops for interviews", () => {
    expect(stopProgress(run("interview", "running", ["anything"]))).toEqual([]);
  });

  it("marks the latest stop current for a queued run that already has events", () => {
    expect(states(run("job_market", "queued", ["search_cached_postings"]))).toEqual(["current", "waiting", "waiting"]);
  });

  it("gives every stop at least one agent step", () => {
    for (const stops of Object.values(STOPS)) {
      for (const stop of stops) expect(stop.nodes?.length, `${stop.id} has no steps`).toBeGreaterThan(0);
    }
  });

  it("maps every background run kind to at least one stop", () => {
    for (const s of STATIONS) {
      for (const kind of s.kinds.filter((k) => k !== "interview")) {
        expect(STOPS[kind as Exclude<Run["kind"], "interview">].length).toBeGreaterThan(0);
      }
    }
  });
});

describe("isFinished", () => {
  it.each([["queued", false], ["running", false], ["succeeded", true], ["failed", true]] as const)(
    "%s -> %s", (status, finished) => expect(isFinished({ status })).toBe(finished),
  );
});

describe("station", () => {
  it("finds a station by id", () => {
    expect(station("market").letter).toBe("M");
  });

  it("throws for an unknown id", () => {
    expect(() => station("nowhere" as never)).toThrow("Unknown station");
  });

  it("gives every station a unique letter", () => {
    const letters = STATIONS.map((s) => s.letter);
    expect(new Set(letters).size).toBe(letters.length);
  });
});
