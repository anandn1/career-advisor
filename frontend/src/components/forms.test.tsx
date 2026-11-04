import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { StartForm } from "@/components/start/start-form";
import { SkillChips } from "@/components/stations/skills-station";
import { jsonResponse, renderWithQuery } from "@/test/render";

const push = vi.fn();
vi.mock("next/navigation", () => ({ useRouter: () => ({ push }) }));

const session = { id: "s1", created_at: "", target_role: "Dev", location: "Pune", github_username: null, skills: [] };

function requests(fetchMock: ReturnType<typeof vi.fn>) {
  return fetchMock.mock.calls.map(([input, init]) => {
    const request = input instanceof Request ? input : new Request(input, init);
    return { method: request.method, path: new URL(request.url).pathname, request };
  });
}

beforeEach(() => push.mockReset());
afterEach(() => vi.unstubAllGlobals());

describe("StartForm", () => {
  async function fill(background: "resume" | "github" | "later") {
    const user = userEvent.setup();
    // Typing is tested on its own below; here the fields only need values.
    await user.click(screen.getByLabelText("Target role"));
    await user.paste("Backend developer");
    await user.click(screen.getByLabelText("Where"));
    await user.paste("Pune");
    await user.click(screen.getByLabelText({ resume: "Upload a resume", github: "Use my GitHub", later: "Add skills later" }[background]));
    return user;
  }

  it("shows the role being typed at the end of the route", async () => {
    renderWithQuery(<StartForm />);
    await userEvent.setup().type(screen.getByLabelText("Target role"), "Data Engineer");
    expect(screen.getByText("Data Engineer")).toBeInTheDocument();
  });

  it("creates a session, remembers it and opens the skills station", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(session, 201));
    vi.stubGlobal("fetch", fetchMock);
    renderWithQuery(<StartForm />);
    const user = await fill("later");
    await user.click(screen.getByRole("button", { name: "Start the route" }));
    await waitFor(() => expect(push).toHaveBeenCalledWith("/s/s1/skills"));
    expect(window.localStorage.getItem("career-advisor:session")).toBe("s1");
    expect(requests(fetchMock).map((r) => `${r.method} ${r.path}`)).toEqual(["POST /api/v1/sessions"]);
  });

  it("starts a GitHub import with the username typed", async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce(jsonResponse(session, 201)).mockResolvedValueOnce(jsonResponse({ id: "r1" }, 202));
    vi.stubGlobal("fetch", fetchMock);
    renderWithQuery(<StartForm />);
    const user = await fill("github");
    await user.type(screen.getByLabelText("Username or profile link"), " octocat ");
    await user.click(screen.getByRole("button", { name: "Start the route" }));
    await waitFor(() => expect(push).toHaveBeenCalled());
    const github = requests(fetchMock)[1];
    expect(github.path).toBe("/api/v1/sessions/s1/runs/skills/github");
    expect(await github.request.json()).toEqual({ github_username: "octocat" });
  });

  it("uploads the chosen resume", async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce(jsonResponse(session, 201)).mockResolvedValueOnce(jsonResponse({ id: "r1" }, 202));
    vi.stubGlobal("fetch", fetchMock);
    renderWithQuery(<StartForm />);
    const user = await fill("resume");
    await user.upload(screen.getByLabelText(/PDF, DOCX or TXT/), new File(["cv"], "cv.txt", { type: "text/plain" }));
    await user.click(screen.getByRole("button", { name: "Start the route" }));
    await waitFor(() => expect(push).toHaveBeenCalled());
    expect(requests(fetchMock)[1].path).toBe("/api/v1/sessions/s1/runs/skills/resume");
  });

  it("shows the server's message and stays on the page when the request is rejected", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ detail: [{ loc: ["body", "location"], msg: "Field required" }] }, 422)));
    renderWithQuery(<StartForm />);
    const user = await fill("later");
    await user.click(screen.getByRole("button", { name: "Start the route" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("location: Field required");
    expect(push).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "Start the route" })).toBeEnabled();
  });

  it("says so when the server cannot be reached", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    renderWithQuery(<StartForm />);
    const user = await fill("later");
    await user.click(screen.getByRole("button", { name: "Start the route" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not reach the server");
  });

  it("reuses the created session when a retry follows a failed import", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse(session, 201))
      .mockResolvedValueOnce(jsonResponse({ detail: "GitHub is busy" }, 503))
      .mockResolvedValueOnce(jsonResponse({ id: "r1" }, 202));
    vi.stubGlobal("fetch", fetchMock);
    renderWithQuery(<StartForm />);
    const user = await fill("github");
    await user.type(screen.getByLabelText("Username or profile link"), "octocat");
    await user.click(screen.getByRole("button", { name: "Start the route" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("GitHub is busy");
    await user.click(screen.getByRole("button", { name: "Start the route" }));
    await waitFor(() => expect(push).toHaveBeenCalledWith("/s/s1/skills"));
    expect(requests(fetchMock).map((r) => `${r.method} ${r.path}`)).toEqual([
      "POST /api/v1/sessions",
      "POST /api/v1/sessions/s1/runs/skills/github",
      "POST /api/v1/sessions/s1/runs/skills/github",
    ]);
  });

  it("updates the created session when the role or place changed before a retry", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse(session, 201))
      .mockResolvedValueOnce(jsonResponse({ detail: "GitHub is busy" }, 503))
      .mockResolvedValueOnce(jsonResponse({ ...session, location: "Remote" }))
      .mockResolvedValueOnce(jsonResponse({ id: "r1" }, 202));
    vi.stubGlobal("fetch", fetchMock);
    renderWithQuery(<StartForm />);
    const user = await fill("github");
    await user.type(screen.getByLabelText("Username or profile link"), "octocat");
    await user.click(screen.getByRole("button", { name: "Start the route" }));
    await screen.findByRole("alert");
    await user.clear(screen.getByLabelText("Where"));
    await user.type(screen.getByLabelText("Where"), "Remote");
    await user.click(screen.getByRole("button", { name: "Start the route" }));
    await waitFor(() => expect(push).toHaveBeenCalledWith("/s/s1/skills"));
    const sent = requests(fetchMock);
    expect(sent.map((r) => `${r.method} ${r.path}`)).toEqual([
      "POST /api/v1/sessions",
      "POST /api/v1/sessions/s1/runs/skills/github",
      "PATCH /api/v1/sessions/s1",
      "POST /api/v1/sessions/s1/runs/skills/github",
    ]);
    expect(await sent[2].request.json()).toEqual({ target_role: "Backend developer", location: "Remote" });
  });

  it("starts no import when the GitHub name is blank", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(session, 201));
    vi.stubGlobal("fetch", fetchMock);
    renderWithQuery(<StartForm />);
    const user = await fill("github");
    await user.type(screen.getByLabelText("Username or profile link"), "   ");
    await user.click(screen.getByRole("button", { name: "Start the route" }));
    await waitFor(() => expect(push).toHaveBeenCalledWith("/s/s1/skills"));
    expect(requests(fetchMock).map((r) => r.path)).toEqual(["/api/v1/sessions"]);
  });

  it("does not upload a file chosen before switching to 'Add skills later'", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(session, 201));
    vi.stubGlobal("fetch", fetchMock);
    renderWithQuery(<StartForm />);
    const user = await fill("resume");
    await user.upload(screen.getByLabelText(/PDF, DOCX or TXT/), new File(["cv"], "cv.txt", { type: "text/plain" }));
    await user.click(screen.getByLabelText("Add skills later"));
    await user.click(screen.getByRole("button", { name: "Start the route" }));
    await waitFor(() => expect(push).toHaveBeenCalledWith("/s/s1/skills"));
    expect(requests(fetchMock).map((r) => r.path)).toEqual(["/api/v1/sessions"]);
  });

  it("asks for a file instead of creating a session when resume is picked without one", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    renderWithQuery(<StartForm />);
    const user = await fill("resume");
    await user.click(screen.getByRole("button", { name: "Start the route" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Choose a resume file, or pick “Add skills later”.");
    expect(fetchMock).not.toHaveBeenCalled();
    expect(push).not.toHaveBeenCalled();
  });

  it("offers to continue a remembered route", async () => {
    window.localStorage.setItem("career-advisor:session", "old-session");
    renderWithQuery(<StartForm />);
    fireEvent.click(screen.getByRole("button", { name: "Continue where you left off" }));
    expect(push).toHaveBeenCalledWith("/s/old-session/skills");
  });

  it("does not offer to continue when nothing is remembered", () => {
    renderWithQuery(<StartForm />);
    expect(screen.queryByRole("button", { name: "Continue where you left off" })).not.toBeInTheDocument();
  });
});

describe("SkillChips", () => {
  async function patchBody(fetchMock: ReturnType<typeof vi.fn>) {
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    const { method, path, request } = requests(fetchMock)[0];
    return { method, path, body: await request.json() };
  }

  it("asks the user to add skills when there are none", () => {
    renderWithQuery(<SkillChips sessionId="s1" skills={[]} />);
    expect(screen.getByText(/No skills yet/)).toBeInTheDocument();
  });

  it("removes a skill", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ ...session, skills: ["SQL"] }));
    vi.stubGlobal("fetch", fetchMock);
    renderWithQuery(<SkillChips sessionId="s1" skills={["Go", "SQL"]} />);
    await userEvent.setup().click(screen.getByRole("button", { name: "Remove Go" }));
    expect(await patchBody(fetchMock)).toEqual({ method: "PATCH", path: "/api/v1/sessions/s1", body: { skills: ["SQL"] } });
  });

  it("adds a trimmed skill and clears the field", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ ...session, skills: ["Go", "Rust"] }));
    vi.stubGlobal("fetch", fetchMock);
    renderWithQuery(<SkillChips sessionId="s1" skills={["Go"]} />);
    const user = userEvent.setup();
    await user.type(screen.getByLabelText("Add a skill"), "  Rust  ");
    await user.click(screen.getByRole("button", { name: "Add skill" }));
    expect((await patchBody(fetchMock)).body).toEqual({ skills: ["Go", "Rust"] });
    expect(screen.getByLabelText("Add a skill")).toHaveValue("");
  });

  // The skipped entry is followed by a real one; only the real one may reach the server.
  it.each([["a blank skill", "   "], ["a skill already listed in another case", "go"]])(
    "sends nothing for %s", async (_, typed) => {
      const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ ...session, skills: ["Go", "Rust"] }));
      vi.stubGlobal("fetch", fetchMock);
      renderWithQuery(<SkillChips sessionId="s1" skills={["Go"]} />);
      const user = userEvent.setup();
      await user.type(screen.getByLabelText("Add a skill"), `${typed}{Enter}`);
      await user.clear(screen.getByLabelText("Add a skill"));
      await user.type(screen.getByLabelText("Add a skill"), "Rust{Enter}");
      await waitFor(() => expect(fetchMock).toHaveBeenCalled());
      expect(fetchMock).toHaveBeenCalledTimes(1);
      expect(await requests(fetchMock)[0].request.json()).toEqual({ skills: ["Go", "Rust"] });
    },
  );

  it("stores the saved session so the page shows it", async () => {
    const saved = { ...session, skills: ["Go", "Rust"] };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(saved)));
    const { client } = renderWithQuery(<SkillChips sessionId="s1" skills={["Go"]} />);
    const user = userEvent.setup();
    await user.type(screen.getByLabelText("Add a skill"), "Rust{Enter}");
    await waitFor(() => expect(client.getQueryData(["session", "s1"])).toEqual(saved));
  });

  it("disables adding while a save is in flight", async () => {
    vi.stubGlobal("fetch", vi.fn(() => new Promise<Response>(() => {})));
    renderWithQuery(<SkillChips sessionId="s1" skills={["Go"]} />);
    const user = userEvent.setup();
    await user.type(screen.getByLabelText("Add a skill"), "Rust{Enter}");
    await user.type(screen.getByLabelText("Add a skill"), "SQL");
    await waitFor(() => expect(screen.getByRole("button", { name: "Add skill" })).toBeDisabled());
  });

  it("cannot add a blank skill", async () => {
    renderWithQuery(<SkillChips sessionId="s1" skills={[]} />);
    await userEvent.setup().type(screen.getByLabelText("Add a skill"), "   ");
    expect(screen.getByRole("button", { name: "Add skill" })).toBeDisabled();
  });

  it("shows the error when saving fails", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ detail: "Session not found or expired" }, 404)));
    renderWithQuery(<SkillChips sessionId="s1" skills={["Go"]} />);
    await userEvent.setup().click(screen.getByRole("button", { name: "Remove Go" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Session not found or expired");
  });
});
