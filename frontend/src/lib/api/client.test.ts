import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, postForm, runEventsUrl, runUrl, unwrap, uploadResume } from "@/lib/api/client";
import { jsonResponse } from "@/test/render";

const response = (status: number) => new Response(null, { status });

describe("unwrap", () => {
  it("returns the data of a successful response", () => {
    expect(unwrap({ data: { id: "s1" }, response: response(200) })).toEqual({ id: "s1" });
  });

  function caught(result: Parameters<typeof unwrap>[0]): ApiError {
    try {
      unwrap(result);
    } catch (error) {
      return error as ApiError;
    }
    throw new Error("unwrap did not throw");
  }

  it("uses the server's string detail as the message", () => {
    const error = caught({ error: { detail: "Session not found or expired" }, response: response(404) });
    expect(error).toBeInstanceOf(ApiError);
    expect([error.status, error.message]).toEqual([404, "Session not found or expired"]);
  });

  it("uses a validation error's message alone when it names only the body", () => {
    expect(() => unwrap({ error: { detail: [{ loc: ["body"], msg: "Expected an object" }] }, response: response(422) }))
      .toThrow(/^Expected an object$/);
  });

  it("says 'Invalid request' for a validation error without field or message", () => {
    expect(() => unwrap({ error: { detail: [{ loc: ["body"] }] }, response: response(422) })).toThrow(/^Invalid request$/);
  });

  it("says 'is invalid' for a named field without a message", () => {
    expect(() => unwrap({ error: { detail: [{ loc: ["body", "skills", 0] }] }, response: response(422) }))
      .toThrow(/^skills\.0: is invalid$/);
  });

  it("treats an empty validation list like no detail", () => {
    expect(() => unwrap({ error: { detail: [] }, response: response(422) })).toThrow(/^Request failed \(422\)$/);
  });

  it("treats an empty string detail like no detail", () => {
    expect(() => unwrap({ error: { detail: "" }, response: response(500) }))
      .toThrow(/^The server ran into a problem\. Try again\.$/);
  });

  it("names the field of a validation error", () => {
    const error = { detail: [{ loc: ["body", "target_role"], msg: "String should have at least 1 character" }] };
    expect(() => unwrap({ error, response: response(422) }))
      .toThrow("target_role: String should have at least 1 character");
  });

  it("falls back to a generic message for server errors without detail", () => {
    expect(() => unwrap({ error: {}, response: response(503) })).toThrow("The server ran into a problem. Try again.");
  });

  it("reports the status for other errors without detail", () => {
    expect(() => unwrap({ error: {}, response: response(418) })).toThrow("Request failed (418)");
  });

  it("treats a non-ok response as an error even without an error body", () => {
    expect(() => unwrap({ data: {}, response: response(500) })).toThrow(ApiError);
  });

  it("keeps the status on the error", () => {
    expect(caught({ error: { detail: "busy" }, response: response(409) }).status).toBe(409);
  });
});

describe("postForm and uploadResume", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("posts the file to the skills endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ id: "r1" }, 202));
    vi.stubGlobal("fetch", fetchMock);
    const run = await uploadResume("s1", new File(["cv"], "cv.txt"), "skills");
    const [url, init] = fetchMock.mock.calls[0];
    expect(run).toEqual({ id: "r1" });
    expect(url).toMatch(/\/api\/v1\/sessions\/s1\/runs\/skills\/resume$/);
    expect((init.body as FormData).get("file")).toBeInstanceOf(File);
    expect((init.body as FormData).has("request")).toBe(false);
    expect(init.method).toBe("POST");
  });

  it("sends the review request text to the review endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ id: "r2" }, 202));
    vi.stubGlobal("fetch", fetchMock);
    await uploadResume("s1", new File(["cv"], "cv.pdf"), "review", "Be blunt");
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toMatch(/\/runs\/resume-review$/);
    expect((init.body as FormData).get("request")).toBe("Be blunt");
  });

  it("raises the server's message when an upload is rejected", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ detail: "Upload one of: .pdf, .docx, .txt" }, 415)));
    await expect(postForm("/x", new FormData())).rejects.toThrow("Upload one of: .pdf, .docx, .txt");
  });

  it("handles an error response that is not JSON", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("Bad gateway", { status: 502 })));
    await expect(postForm("/x", new FormData())).rejects.toThrow("The server ran into a problem. Try again.");
  });
});

describe("URLs", () => {
  it("builds resume file and event stream URLs", () => {
    expect(runUrl("s1", "r1", "resume.pdf")).toMatch(/\/api\/v1\/sessions\/s1\/runs\/r1\/resume\.pdf$/);
    expect(runEventsUrl("s1", "r1")).toMatch(/\/api\/v1\/sessions\/s1\/runs\/r1\/events$/);
  });
});
