import { render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { RouteBar } from "@/components/transit/route-bar";
import { RunLine } from "@/components/transit/run-line";
import type { Run } from "@/lib/api/client";
import { station } from "@/lib/stations";

vi.mock("next/link", () => ({
  default: ({ href, children, ...rest }: { href: string; children: React.ReactNode }) => (
    <a href={href} {...rest}>{children}</a>
  ),
}));

function run(status: Run["status"], steps: string[], error: string | null = null): Run {
  return {
    id: "r1", kind: "job_market", status, error, result: null, created_at: "", started_at: null, finished_at: null,
    events: steps.map((step) => ({ step, detail: {}, at: "" })),
  };
}

describe("RunLine", () => {
  const market = station("market");

  it("describes each stop's state for screen readers", () => {
    render(<RunLine run={run("running", ["search_cached_postings", "scrape_fresh_postings"])} station={market} />);
    const stops = within(screen.getByRole("list", { name: "Market progress" })).getAllByRole("listitem");
    expect(stops.map((stop) => stop.textContent)).toEqual([
      "Saved postingsdone",
      "Live LinkedIn postingsin progress",
      "Skill gapsnot reached",
      "Arrival",
    ]);
  });

  it("says Arrived when the run succeeded", () => {
    render(<RunLine run={run("succeeded", ["analyze_skill_gaps"])} station={market} />);
    expect(screen.getByText("Arrived")).toBeInTheDocument();
  });

  it("says Stopped when the run failed", () => {
    render(<RunLine run={run("failed", [])} station={market} />);
    expect(screen.getByText("Stopped")).toBeInTheDocument();
  });

  it("shows how often a looping stop was entered", () => {
    const steps = ["iteration_controller", "start_evaluation", "iteration_controller"];
    render(<RunLine run={{ ...run("running", steps), kind: "learning_path" }} station={station("path")} />);
    expect(screen.getByText("×2")).toBeInTheDocument();
  });
});

describe("RouteBar", () => {
  it("links every station and the report under the session", () => {
    render(<RouteBar sessionId="s1" current="skills" status={{}} terminus="Backend Developer" />);
    const hrefs = screen.getAllByRole("link").map((link) => link.getAttribute("href"));
    expect(hrefs).toEqual(["/s/s1/skills", "/s/s1/market", "/s/s1/path", "/s/s1/resume", "/s/s1/interview", "/s/s1/report"]);
  });

  it("marks the station being viewed as the current page", () => {
    render(<RouteBar sessionId="s1" current="path" status={{}} terminus="Dev" />);
    expect(screen.getByRole("link", { current: "page" })).toHaveTextContent("Learning path");
  });

  it("announces each station's status", () => {
    render(<RouteBar sessionId="s1" current="report" status={{ skills: "done", market: "running" }} terminus="Dev" />);
    expect(screen.getByRole("link", { name: /Skills/ })).toHaveTextContent("(has results)");
    expect(screen.getByRole("link", { name: /Market/ })).toHaveTextContent("(running)");
    expect(screen.getByRole("link", { name: /Interview/ })).toHaveTextContent("(not started)");
  });

  it("names the target role at the end of the line", () => {
    render(<RouteBar sessionId="s1" current="report" status={{}} terminus="Data Engineer" />);
    expect(screen.getByRole("link", { current: "page" })).toHaveTextContent("Data Engineer");
  });
});
