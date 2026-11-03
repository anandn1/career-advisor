import createClient from "openapi-fetch";

import type { components, paths } from "./schema";

export type Session = components["schemas"]["SessionOut"];
export type Run = components["schemas"]["RunOut"];
export type RunKind = components["schemas"]["RunKind"];
export type RunStatus = components["schemas"]["RunStatus"];
export type Interview = components["schemas"]["InterviewOut"];
export type Turn = components["schemas"]["TurnOut"];
export type Template = components["schemas"]["TemplateOut"];
export type ResumeDraft = components["schemas"]["ResumeDraft"];

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// Look fetch up on each call (not once at import), so it can be replaced in tests.
export const api = createClient<paths>({ baseUrl: API_URL, fetch: (request) => globalThis.fetch(request) });

/** An API failure carrying the server's own explanation (FastAPI's `detail`). */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

type FastApiDetail = string | { msg?: string; loc?: (string | number)[] }[] | undefined;

function describe(detail: FastApiDetail, status: number): string {
  if (typeof detail === "string" && detail) return detail;
  if (Array.isArray(detail) && detail.length > 0) {
    const first = detail[0];
    const field = first.loc?.filter((part) => part !== "body").join(".");
    return field ? `${field}: ${first.msg ?? "is invalid"}` : (first.msg ?? "Invalid request");
  }
  return status >= 500 ? "The server ran into a problem. Try again." : `Request failed (${status})`;
}

/** Unwrap an openapi-fetch result: return its data or throw an ApiError. */
export function unwrap<T>(result: { data?: T; error?: unknown; response: Response }): T {
  if (result.error !== undefined || !result.response.ok) {
    const detail = (result.error as { detail?: FastApiDetail } | undefined)?.detail;
    throw new ApiError(result.response.status, describe(detail, result.response.status));
  }
  return result.data as T;
}

export function runUrl(sessionId: string, runId: string, file: "resume.tex" | "resume.pdf") {
  return `${API_URL}/api/v1/sessions/${sessionId}/runs/${runId}/${file}`;
}

export function runEventsUrl(sessionId: string, runId: string) {
  return `${API_URL}/api/v1/sessions/${sessionId}/runs/${runId}/events`;
}

/** POST multipart form data (file uploads), with the same error handling as `unwrap`. */
export async function postForm<T>(path: string, form: FormData): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, { method: "POST", body: form });
  const body = await response.json().catch(() => undefined);
  return unwrap<T>({ data: body, error: response.ok ? undefined : body, response });
}

export function uploadResume(sessionId: string, file: File, kind: "skills" | "review", request = "") {
  const form = new FormData();
  form.append("file", file);
  if (kind === "review") form.append("request", request);
  const endpoint = kind === "skills" ? "skills/resume" : "resume-review";
  return postForm<Run>(`/api/v1/sessions/${sessionId}/runs/${endpoint}`, form);
}
