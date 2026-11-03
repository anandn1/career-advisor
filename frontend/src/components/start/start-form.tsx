"use client";

import { useRouter } from "next/navigation";
import { useId, useState, useSyncExternalStore } from "react";

import { StationRoundel } from "@/components/transit/station-roundel";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { rememberedSession, rememberSession } from "@/hooks/use-career";
import { api, ApiError, unwrap, uploadResume } from "@/lib/api/client";
import { STATIONS } from "@/lib/stations";

type Background = "resume" | "github" | "later";

const noSubscription = () => () => {};

/** The opening route: from "You" to whatever role is being typed. */
function OpeningRoute({ role }: { role: string }) {
  return (
    <div aria-hidden className="flex w-full min-w-0 items-center sm:w-auto">
      <span className="signage mr-2 shrink-0 text-sm text-ink-muted sm:mr-3">You</span>
      {STATIONS.map((station) => (
        <span key={station.id} className="flex flex-1 items-center sm:flex-none">
          <span className="h-[7px] min-w-1.5 flex-1 sm:w-12 sm:flex-none" style={{ background: station.colour }} />
          <StationRoundel station={station} size="sm" />
        </span>
      ))}
      <span className="h-[7px] min-w-1.5 flex-1 bg-ink sm:w-12 sm:flex-none" />
      <span className="min-w-0 truncate rounded-md bg-ink px-2.5 py-1.5 font-heading text-sm font-extrabold whitespace-nowrap text-surface sm:px-3 sm:text-lg">
        {role.trim() || "Your next role"}
      </span>
    </div>
  );
}

async function createSession(role: string, location: string) {
  return unwrap(await api.POST("/api/v1/sessions", { body: { target_role: role, location } }));
}

async function updateSession(sessionId: string, role: string, location: string) {
  unwrap(
    await api.PATCH("/api/v1/sessions/{session_id}", {
      params: { path: { session_id: sessionId } },
      body: { target_role: role, location },
    }),
  );
}

type CreatedSession = { id: string; role: string; location: string };

/** The session to import into: a new one, or the one an earlier attempt created, kept up to date. */
async function sessionFor(created: CreatedSession | null, role: string, location: string): Promise<string> {
  if (!created) return (await createSession(role, location)).id;
  if (created.role !== role || created.location !== location) await updateSession(created.id, role, location);
  return created.id;
}

async function importBackground(sessionId: string, background: Background, file: File | null, github: string) {
  if (background === "resume" && file) {
    await uploadResume(sessionId, file, "skills");
  } else if (background === "github" && github.trim()) {
    unwrap(
      await api.POST("/api/v1/sessions/{session_id}/runs/skills/github", {
        params: { path: { session_id: sessionId } },
        body: { github_username: github.trim() },
      }),
    );
  }
}

export function StartForm() {
  const router = useRouter();
  const ids = { role: useId(), location: useId(), file: useId(), github: useId() };
  const [role, setRole] = useState("");
  const [location, setLocation] = useState("");
  const [background, setBackground] = useState<Background>("resume");
  const [file, setFile] = useState<File | null>(null);
  const [github, setGithub] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  // localStorage only exists in the browser; the server snapshot is "no previous route".
  const previous = useSyncExternalStore(noSubscription, rememberedSession, () => null);

  // Once created, a retry reuses this session (updated if the role or place changed)
  // instead of creating, and orphaning, another.
  const [created, setCreated] = useState<CreatedSession | null>(null);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (background === "resume" && !file) {
      setError("Choose a resume file, or pick “Add skills later”.");
      return;
    }
    setError(null);
    setSubmitting(true);
    try {
      const sessionId = await sessionFor(created, role, location);
      setCreated({ id: sessionId, role, location });
      rememberSession(sessionId);
      await importBackground(sessionId, background, file, github);
      router.push(`/s/${sessionId}/skills`);
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "Could not reach the server. Is the API running?");
      setSubmitting(false);
    }
  }

  return (
    <div className="flex flex-col gap-10">
      <header className="flex flex-col gap-6">
        <p className="signage text-sm text-ink-muted">Agentic AI Career Advisor</p>
        <h1 className="max-w-3xl font-heading text-4xl font-extrabold leading-[1.05] tracking-tight sm:text-6xl">
          Plan the route from the skills you have to the job you want.
        </h1>
        <OpeningRoute role={role} />
        <p className="max-w-2xl text-lg text-ink-muted">
          Five AI agents read live job postings, find what you are missing, pick courses, write a
          resume for the role and interview you on your weak spots.
        </p>
      </header>

      <form onSubmit={submit} className="grid gap-6 rounded-xl border-2 border-ink bg-surface p-6 sm:grid-cols-2 sm:p-8">
        <div className="flex flex-col gap-2">
          <Label htmlFor={ids.role} className="signage text-xs">Target role</Label>
          <Input id={ids.role} required maxLength={200} value={role} onChange={(e) => setRole(e.target.value)}
                 placeholder="Backend developer" autoComplete="organization-title" />
        </div>
        <div className="flex flex-col gap-2">
          <Label htmlFor={ids.location} className="signage text-xs">Where</Label>
          <Input id={ids.location} required maxLength={200} value={location} onChange={(e) => setLocation(e.target.value)}
                 placeholder="Pune, or Remote" />
        </div>

        <fieldset className="flex flex-col gap-3 sm:col-span-2">
          <legend className="signage mb-2 text-xs">Your background</legend>
          <div className="flex flex-wrap gap-2" role="radiogroup">
            {([["resume", "Upload a resume"], ["github", "Use my GitHub"], ["later", "Add skills later"]] as const).map(
              ([value, label]) => (
                <label key={value}
                       className="flex cursor-pointer items-center gap-2 rounded-full border-2 border-rule px-3 py-1.5 has-checked:border-ink has-checked:bg-paper">
                  <input type="radio" name="background" value={value} checked={background === value}
                         onChange={() => setBackground(value)} className="accent-[var(--color-ink)]" />
                  {label}
                </label>
              ),
            )}
          </div>
          {background === "resume" && (
            <div className="flex flex-col gap-2">
              <Label htmlFor={ids.file} className="text-sm text-ink-muted">PDF, DOCX or TXT, up to 5 MB. Only the text is kept.</Label>
              <Input id={ids.file} type="file" accept=".pdf,.docx,.txt" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
            </div>
          )}
          {background === "github" && (
            <div className="flex flex-col gap-2">
              <Label htmlFor={ids.github} className="text-sm text-ink-muted">Username or profile link</Label>
              <Input id={ids.github} value={github} onChange={(e) => setGithub(e.target.value)} placeholder="octocat" />
            </div>
          )}
        </fieldset>

        {error && <p role="alert" className="text-danger sm:col-span-2">{error}</p>}

        <div className="flex flex-wrap items-center gap-4 sm:col-span-2">
          <Button type="submit" size="lg" disabled={submitting} className="font-heading font-bold">
            {submitting ? "Starting…" : "Start the route"}
          </Button>
          {previous && (
            <Button type="button" variant="link" onClick={() => router.push(`/s/${previous}/skills`)}>
              Continue where you left off
            </Button>
          )}
        </div>
      </form>
    </div>
  );
}
