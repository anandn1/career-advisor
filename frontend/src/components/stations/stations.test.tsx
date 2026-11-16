import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { SessionShell } from "@/components/session/session-shell";
import { InterviewStation } from "@/components/stations/interview-station";
import { MarketResults } from "@/components/stations/market-station";
import { GapsWithoutCourses, PathResults } from "@/components/stations/path-station";
import { jsonResponse, renderWithQuery } from "@/test/render";

vi.mock("next/navigation", () => ({ useParams: () => ({ sessionId: "s1" }), usePathname: () => "/s/s1/interview" }));
vi.mock("next/link", () => ({
  default: ({ href, children, ...rest }: { href: string; children: React.ReactNode }) => <a href={href} {...rest}>{children}</a>,
}));

const session = { id: "s1", created_at: "", target_role: "Dev", location: "Pune", github_username: null, skills: [] };
const report = { session, job_market: null, learning_path: null, resume: null, resume_review: null, interviews: [] };
const TOPICS: Record<string, { topics: string[]; suggested: string[] }> = {
  generic: { topics: ["Databases", "Networking", "System Design"], suggested: ["Databases", "System Design"] },
  google: { topics: ["Distributed Systems", "System Design"], suggested: ["System Design"] },
};
const started = {
  id: "i1", status: "running", company_focus: "generic", topics: ["System Design"], report: [],
  turn: { question: "Design a cache.", feedback: null, finished: false, summary: null, question_count: 1 },
};

afterEach(() => vi.unstubAllGlobals());

type Options = { topicsStatus?: number; topics?: typeof TOPICS; topicsPending?: boolean; start?: () => Response };

/** The API, with topics per ?company_focus=; the options fail, hold or replace answers. */
function serve({ topicsStatus = 200, topics = TOPICS, topicsPending = false, start }: Options = {}) {
  const fetchMock = vi.fn(async (input: Request) => {
    const url = new URL(input.url);
    if (url.pathname.endsWith("/interview-topics") && topicsPending) return new Promise<Response>(() => {});
    const routes: Record<string, () => Response> = {
      "/api/v1/sessions/s1": () => jsonResponse(session),
      "/api/v1/sessions/s1/report": () => jsonResponse(report),
      "/api/v1/sessions/s1/runs": () => jsonResponse([]),
      "/api/v1/sessions/s1/interviews": start ?? (() => jsonResponse(started, 201)),
      "/api/v1/sessions/s1/interview-topics": () =>
        topicsStatus === 200
          ? jsonResponse(topics[url.searchParams.get("company_focus") ?? ""])
          : jsonResponse({ detail: "Question bank unavailable" }, topicsStatus),
    };
    return routes[url.pathname]?.() ?? jsonResponse({ detail: "Not Found" }, 404);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

async function startBody(fetchMock: ReturnType<typeof serve>) {
  await waitFor(() => expect(fetchMock.mock.calls.some(([r]) => (r as Request).method === "POST")).toBe(true));
  const [request] = fetchMock.mock.calls.find(([r]) => (r as Request).method === "POST")!;
  return (request as Request).json();
}

const pressed = () =>
  screen.getAllByRole("button", { pressed: true }).map((button) => button.textContent);

async function renderInterview() {
  renderWithQuery(<SessionShell><InterviewStation /></SessionShell>);
  await screen.findByRole("button", { name: "Networking" });
}

describe("Interview topics", () => {
  it("offers the company's topics with the ones for the skill gaps chosen", async () => {
    serve();
    await renderInterview();
    const topics = within(screen.getByRole("group", { name: "Topics" })).getAllByRole("button");
    expect(topics.map((button) => button.textContent)).toEqual(["Databases", "Networking", "System Design"]);
    expect(pressed()).toEqual(["Databases", "System Design"]);
    expect(screen.getByText(/Picked from your skill gaps/)).toBeInTheDocument();
  });

  it("starts the interview on the topics chosen", async () => {
    const fetchMock = serve();
    await renderInterview();
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Networking" }));
    await user.click(screen.getByRole("button", { name: "Databases" }));
    expect(screen.getByText("2 chosen.")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Start the interview" }));
    expect(await startBody(fetchMock)).toEqual({ company_focus: "generic", topics: ["System Design", "Networking"] });
  });

  it("leaves the topics to the server when none are chosen", async () => {
    const fetchMock = serve();
    await renderInterview();
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Databases" }));
    await user.click(screen.getByRole("button", { name: "System Design" }));
    expect(screen.getByText(/None chosen/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Start the interview" }));
    expect(await startBody(fetchMock)).toEqual({ company_focus: "generic", topics: null });
  });

  it("shows another company's own topics and suggestions, dropping earlier picks", async () => {
    const fetchMock = serve();
    await renderInterview();
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Networking" }));
    await user.selectOptions(screen.getByLabelText("Company questions"), "google");
    await screen.findByRole("button", { name: "Distributed Systems" });
    expect(screen.queryByRole("button", { name: "Networking" })).not.toBeInTheDocument();
    expect(pressed()).toEqual(["System Design"]);
    await user.click(screen.getByRole("button", { name: "Start the interview" }));
    expect(await startBody(fetchMock)).toEqual({ company_focus: "google", topics: ["System Design"] });
  });

  it("says why the topics could not be loaded", async () => {
    serve({ topicsStatus: 503 });
    renderWithQuery(<SessionShell><InterviewStation /></SessionShell>);
    expect(await screen.findByRole("alert")).toHaveTextContent("Question bank unavailable");
  });

  it("waits for the topics before the interview can start", async () => {
    serve({ topicsPending: true });
    renderWithQuery(<SessionShell><InterviewStation /></SessionShell>);
    expect(await screen.findByText("Loading this company's topics…")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Start the interview" })).toBeDisabled();
  });

  it("never sends a suggested topic the company does not offer", async () => {
    const fetchMock = serve({ topics: { generic: { topics: TOPICS.generic.topics, suggested: ["Ghost", "System Design"] } } });
    await renderInterview();
    expect(pressed()).toEqual(["System Design"]);
    await userEvent.setup().click(screen.getByRole("button", { name: "Start the interview" }));
    expect(await startBody(fetchMock)).toEqual({ company_focus: "generic", topics: ["System Design"] });
  });

  it("says why the interview could not start", async () => {
    serve({ start: () => jsonResponse({ detail: "No generic questions on: Python." }, 422) });
    await renderInterview();
    await userEvent.setup().click(screen.getByRole("button", { name: "Start the interview" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("No generic questions on: Python.");
  });
});

describe("PathResults", () => {
  const path = {
    user_summary: "A backend developer.",
    skill_gaps: ["go"],
    recommendations: [{ skill_name: "go", course_title: "Go Tour", course_url: "https://go.dev/tour", description: "Basics." }],
  };

  it("lists the gaps without a course under the path, with the web search hint", () => {
    render(<PathResults result={{ path, gaps_without_courses: ["rust"], web_search_enabled: false }} />);
    expect(screen.getByRole("link", { name: "Go Tour" })).toBeInTheDocument();
    expect(screen.getByText(/No course found for:/).parentElement).toHaveTextContent("No course found for: rust.");
    expect(screen.getByText(/Web search for courses is off/)).toBeInTheDocument();
  });

  it("shows a path planned before gaps without courses were reported as it was", () => {
    render(<PathResults result={{ path }} />);
    expect(screen.getByRole("link", { name: "Go Tour" })).toBeInTheDocument();
    expect(screen.queryByText(/No course found/)).not.toBeInTheDocument();
  });
});

describe("GapsWithoutCourses", () => {
  it("lists the gaps no course was found for", () => {
    render(<GapsWithoutCourses gaps={["kubernetes", "go"]} webSearch={true} />);
    expect(screen.getByText(/No course found for:/).parentElement).toHaveTextContent("No course found for: kubernetes, go.");
    expect(screen.queryByText(/TAVILY_API_KEY/)).not.toBeInTheDocument();
  });

  it("says how to search the web when web search was off", () => {
    render(<GapsWithoutCourses gaps={["go"]} webSearch={false} />);
    expect(screen.getByText(/Web search for courses is off/)).toHaveTextContent("Set TAVILY_API_KEY to search the web as well.");
  });

  it("gives no hint for paths planned before web search was reported", () => {
    render(<GapsWithoutCourses gaps={["go"]} webSearch={undefined} />);
    expect(screen.queryByText(/Web search for courses is off/)).not.toBeInTheDocument();
  });

  it("shows nothing when every gap has a course", () => {
    const { container } = render(<GapsWithoutCourses gaps={[]} webSearch={false} />);
    expect(container).toBeEmptyDOMElement();
  });
});

describe("MarketResults", () => {
  const job = { title: "Backend Engineer", company: "Acme", location: "Pune", skills_required: ["go"] };

  it("links each posting's title to it on LinkedIn, in a new tab", () => {
    const url = "https://in.linkedin.com/jobs/view/1";
    render(<MarketResults result={{ jobs: [{ ...job, url }], analysis: {} }} />);
    const link = screen.getByRole("link", { name: "Backend Engineer (opens on LinkedIn)" });
    expect(link).toHaveAttribute("href", url);
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noreferrer");
  });

  it("shows postings without a link as plain titles", () => {
    render(<MarketResults result={{ jobs: [{ ...job, url: null }, { ...job, title: "Older posting" }], analysis: {} }} />);
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
    expect(screen.getByText("Older posting")).toBeInTheDocument();
  });
});
