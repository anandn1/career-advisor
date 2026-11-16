import { act, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { RunPanel } from "@/components/session/run-panel";
import { SessionShell, useCurrentSession } from "@/components/session/session-shell";
import type { Run } from "@/lib/api/client";
import { station } from "@/lib/stations";
import { FakeEventSource } from "@/test/fake-event-source";
import { jsonResponse, renderWithQuery } from "@/test/render";

let pathname = "/s/s1/market";
vi.mock("next/navigation", () => ({ useParams: () => ({ sessionId: "s1" }), usePathname: () => pathname }));
vi.mock("next/link", () => ({
  default: ({ href, children, ...rest }: { href: string; children: React.ReactNode }) => <a href={href} {...rest}>{children}</a>,
}));

const session = { id: "s1", created_at: "", target_role: "Backend developer", location: "Pune", github_username: null, skills: [] as string[] };
const emptyReport = { session, job_market: null, learning_path: null, resume: null, resume_review: null, interviews: [] };

function makeRun(kind: Run["kind"], status: Run["status"], error: string | null = null): Run {
  return { id: "r1", kind, status, error, result: { ok: true }, created_at: "", started_at: null, finished_at: null, events: [] };
}

/** Answer each API path from a table; anything unlisted is a 404. */
function serve(routes: Record<string, () => Response>) {
  const fetchMock = vi.fn(async (input: Request) => {
    const path = new URL(input.url).pathname;
    return routes[path]?.() ?? jsonResponse({ detail: "Not Found" }, 404);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function shellRoutes(overrides: Record<string, () => Response> = {}) {
  return {
    "/api/v1/sessions/s1": () => jsonResponse(session),
    "/api/v1/sessions/s1/report": () => jsonResponse(emptyReport),
    "/api/v1/sessions/s1/runs": () => jsonResponse([]),
    ...overrides,
  };
}

function SessionName() {
  return <p>Inside: {useCurrentSession().target_role}</p>;
}

const stationLink = (name: string) => screen.getByRole("link", { name: new RegExp(name) });

beforeEach(() => {
  pathname = "/s/s1/market";
  FakeEventSource.instances = [];
  vi.stubGlobal("EventSource", FakeEventSource);
});
afterEach(() => vi.unstubAllGlobals());

describe("SessionShell", () => {
  it("gives the page the session and remembers it", async () => {
    serve(shellRoutes());
    renderWithQuery(<SessionShell><SessionName /></SessionShell>);
    expect(await screen.findByText("Inside: Backend developer")).toBeInTheDocument();
    await waitFor(() => expect(window.localStorage.getItem("career-advisor:session")).toBe("s1"));
  });

  it("says the route expired and forgets it when the session is gone", async () => {
    window.localStorage.setItem("career-advisor:session", "s1");
    serve(shellRoutes({ "/api/v1/sessions/s1": () => jsonResponse({ detail: "Session not found or expired" }, 404) }));
    renderWithQuery(<SessionShell><SessionName /></SessionShell>);
    expect(await screen.findByRole("heading", { name: "This route has expired" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Start a new route" })).toHaveAttribute("href", "/");
    expect(screen.queryByText(/Inside:/)).not.toBeInTheDocument();
    await waitFor(() => expect(window.localStorage.getItem("career-advisor:session")).toBeNull());
  });

  it("shows the error and keeps the remembered route when the server fails", async () => {
    window.localStorage.setItem("career-advisor:session", "s1");
    serve(shellRoutes({ "/api/v1/sessions/s1": () => jsonResponse({ detail: "Database is down" }, 503) }));
    renderWithQuery(<SessionShell><SessionName /></SessionShell>);
    expect(await screen.findByRole("heading", { name: "Can't reach the server" })).toBeInTheDocument();
    expect(screen.getByText(/Database is down/)).toBeInTheDocument();
    expect(window.localStorage.getItem("career-advisor:session")).toBe("s1");
  });

  it("marks the station of an unfinished background run as running", async () => {
    serve(shellRoutes({ "/api/v1/sessions/s1/runs": () => jsonResponse([makeRun("learning_path", "running")]) }));
    renderWithQuery(<SessionShell><SessionName /></SessionShell>);
    await waitFor(() => expect(stationLink("Learning path")).toHaveAccessibleName(/running/i));
    expect(stationLink("Market")).not.toHaveAccessibleName(/running/i);
  });

  it("never marks the interview station as running", async () => {
    serve(shellRoutes({ "/api/v1/sessions/s1/runs": () => jsonResponse([makeRun("interview", "running")]) }));
    renderWithQuery(<SessionShell><SessionName /></SessionShell>);
    await screen.findByText("Inside: Backend developer");
    await waitFor(() => expect(stationLink("Interview")).toHaveAccessibleName(/not started/i));
  });

  it("marks stations with results as done", async () => {
    const report = { ...emptyReport, job_market: { run_id: "m", jobs: [], analysis: {} } };
    serve(shellRoutes({ "/api/v1/sessions/s1/report": () => jsonResponse(report) }));
    renderWithQuery(<SessionShell><SessionName /></SessionShell>);
    await waitFor(() => expect(stationLink("Market")).toHaveAccessibleName(/has results/i));
  });

  it("marks the resume station done from a review alone", async () => {
    const review = { run_id: "v", review: { summary: "", strengths: [], weaknesses: [], suggestions: [] } };
    serve(shellRoutes({ "/api/v1/sessions/s1/report": () => jsonResponse({ ...emptyReport, resume_review: review }) }));
    renderWithQuery(<SessionShell><SessionName /></SessionShell>);
    await waitFor(() => expect(stationLink("Resume")).toHaveAccessibleName(/has results/i));
  });

  it("marks the interview station done only for a finished interview", async () => {
    const interviews = [{ id: "i", status: "running", summary: null }];
    serve(shellRoutes({ "/api/v1/sessions/s1/report": () => jsonResponse({ ...emptyReport, interviews }) }));
    renderWithQuery(<SessionShell><SessionName /></SessionShell>);
    await screen.findByText("Inside: Backend developer");
    await waitFor(() => expect(stationLink("Interview")).toHaveAccessibleName(/not started/i));
  });

  it("marks the skills station done once the session has skills", async () => {
    serve(shellRoutes({ "/api/v1/sessions/s1": () => jsonResponse({ ...session, skills: ["Go"] }) }));
    renderWithQuery(<SessionShell><SessionName /></SessionShell>);
    await waitFor(() => expect(stationLink("Skills")).toHaveAccessibleName(/has results/i));
  });

  describe("polling the run list", () => {
    beforeEach(() => vi.useFakeTimers({ shouldAdvanceTime: true }));
    afterEach(() => vi.useRealTimers());

    const runListFetches = (fetchMock: ReturnType<typeof serve>) =>
      fetchMock.mock.calls.filter(([request]) => new URL(request.url).pathname === "/api/v1/sessions/s1/runs").length;

    it("checks again every 3 seconds while a run is unfinished", async () => {
      const fetchMock = serve(shellRoutes({ "/api/v1/sessions/s1/runs": () => jsonResponse([makeRun("job_market", "running")]) }));
      renderWithQuery(<SessionShell><SessionName /></SessionShell>);
      await waitFor(() => expect(runListFetches(fetchMock)).toBe(1));
      await act(() => vi.advanceTimersByTimeAsync(3_000));
      await waitFor(() => expect(runListFetches(fetchMock)).toBe(2));
    });

    it("stops checking when no run is unfinished", async () => {
      const fetchMock = serve(shellRoutes({ "/api/v1/sessions/s1/runs": () => jsonResponse([makeRun("job_market", "succeeded")]) }));
      renderWithQuery(<SessionShell><SessionName /></SessionShell>);
      await waitFor(() => expect(runListFetches(fetchMock)).toBe(1));
      await act(() => vi.advanceTimersByTimeAsync(10_000));
      expect(runListFetches(fetchMock)).toBe(1);
    });
  });

  it.each([["/s/s1/path", "Learning path"], ["/s/s1/report", "Backend developer"], ["/s/s1", "Backend developer"]])(
    "marks %s as the page being viewed", async (path, name) => {
      pathname = path;
      serve(shellRoutes());
      renderWithQuery(<SessionShell><SessionName /></SessionShell>);
      await screen.findByText("Inside: Backend developer");
      const current = within(screen.getByRole("navigation")).getAllByRole("link")
        .filter((link) => link.getAttribute("aria-current") === "page");
      expect(current.map((link) => link.textContent)).toEqual([expect.stringContaining(name)]);
    },
  );
});

describe("RunPanel", () => {
  const market = station("market");

  function renderPanel(run: Run | null) {
    serve({ "/api/v1/sessions/s1/runs/r1": () => (run ? jsonResponse(run) : jsonResponse({ detail: "Run not found" }, 404)) });
    return renderWithQuery(
      <RunPanel sessionId="s1" station={market} runId={run ? "r1" : undefined}>
        {(finished) => <p>Result of {finished.id}</p>}
      </RunPanel>,
    );
  }

  it("shows nothing before there is a run", () => {
    const { container } = renderPanel(null);
    expect(container).toBeEmptyDOMElement();
  });

  it("shows the line, not the result, while the run travels", async () => {
    renderPanel(makeRun("job_market", "running"));
    expect(await screen.findByText("On the line now")).toBeInTheDocument();
    expect(screen.queryByText("Result of r1")).not.toBeInTheDocument();
  });

  it("shows only the result once the run succeeded", async () => {
    renderPanel(makeRun("job_market", "succeeded"));
    expect(await screen.findByText("Result of r1")).toBeInTheDocument();
    expect(screen.queryByText("On the line now")).not.toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("shows the run's error and no result when it failed", async () => {
    renderPanel(makeRun("job_market", "failed", "No postings found for that role."));
    expect(await screen.findByRole("alert")).toHaveTextContent("No postings found for that role.");
    expect(screen.getByText("Last run")).toBeInTheDocument();
    expect(screen.queryByText("Result of r1")).not.toBeInTheDocument();
  });

  it("says so when the run cannot be loaded", async () => {
    serve({ "/api/v1/sessions/s1/runs/r1": () => jsonResponse({ detail: "Database is down" }, 503) });
    renderWithQuery(<RunPanel sessionId="s1" station={market} runId="r1" />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Couldn't load this run: Database is down");
  });

  it("falls back to a generic message for a failure without an error", async () => {
    renderPanel(makeRun("job_market", "failed"));
    expect(await screen.findByRole("alert")).toHaveTextContent("The agent stopped. Try again.");
  });
});
