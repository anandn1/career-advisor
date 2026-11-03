"use client";

import { X } from "lucide-react";
import { useId, useState } from "react";

import { RunPanel } from "@/components/session/run-panel";
import { useCurrentSession } from "@/components/session/session-shell";
import { StationHeader } from "@/components/session/station-header";
import { useLatestRun } from "@/components/stations/use-latest-run";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useStartRun, useUpdateSession } from "@/hooks/use-career";
import { api, ApiError, unwrap, uploadResume } from "@/lib/api/client";
import { station } from "@/lib/stations";

const SKILLS = station("skills");

export function SkillChips({ sessionId, skills }: { sessionId: string; skills: string[] }) {
  const update = useUpdateSession(sessionId);
  const [draft, setDraft] = useState("");
  const inputId = useId();
  const save = (next: string[]) => update.mutate({ skills: next });

  return (
    <div className="flex flex-col gap-3">
      {skills.length === 0 ? (
        <p className="text-ink-muted">No skills yet. Import them below or add them one at a time.</p>
      ) : (
        <ul className="flex flex-wrap gap-2" aria-label="Your skills">
          {skills.map((skill) => (
            <li key={skill} className="flex items-center gap-1 rounded-full border-2 bg-surface py-1 pr-1 pl-3"
                style={{ borderColor: SKILLS.colour }}>
              <span>{skill}</span>
              <button type="button" onClick={() => save(skills.filter((s) => s !== skill))}
                      className="rounded-full p-1 text-ink-muted hover:bg-paper hover:text-ink"
                      aria-label={`Remove ${skill}`}>
                <X className="size-3.5" />
              </button>
            </li>
          ))}
        </ul>
      )}
      <form
        className="flex max-w-md items-end gap-2"
        onSubmit={(event) => {
          event.preventDefault();
          if (draft.trim()) save([...skills, draft.trim()]);
          setDraft("");
        }}
      >
        <div className="flex flex-1 flex-col gap-1.5">
          <Label htmlFor={inputId} className="signage text-xs">Add a skill</Label>
          <Input id={inputId} value={draft} maxLength={100} onChange={(e) => setDraft(e.target.value)} placeholder="Kubernetes" />
        </div>
        <Button type="submit" variant="outline" disabled={!draft.trim() || update.isPending}>Add skill</Button>
      </form>
      {update.error && <p role="alert" className="text-danger">{update.error.message}</p>}
    </div>
  );
}

function ImportSkills({ sessionId }: { sessionId: string }) {
  const [file, setFile] = useState<File | null>(null);
  const [github, setGithub] = useState("");
  const ids = { file: useId(), github: useId() };
  const fromResume = useStartRun(sessionId, () => uploadResume(sessionId, file!, "skills"));
  const fromGithub = useStartRun(sessionId, async () =>
    unwrap(await api.POST("/api/v1/sessions/{session_id}/runs/skills/github", {
      params: { path: { session_id: sessionId } }, body: { github_username: github.trim() },
    })));
  const error = fromResume.error ?? fromGithub.error;

  return (
    <div className="grid gap-4 sm:grid-cols-2">
      <form className="flex flex-col gap-2 rounded-xl border-2 border-rule bg-surface p-5"
            onSubmit={(e) => { e.preventDefault(); fromResume.mutate(undefined); }}>
        <Label htmlFor={ids.file} className="signage text-xs">From a resume</Label>
        <Input id={ids.file} type="file" accept=".pdf,.docx,.txt" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        <Button type="submit" disabled={!file || fromResume.isPending} className="self-start">Read my resume</Button>
      </form>
      <form className="flex flex-col gap-2 rounded-xl border-2 border-rule bg-surface p-5"
            onSubmit={(e) => { e.preventDefault(); fromGithub.mutate(undefined); }}>
        <Label htmlFor={ids.github} className="signage text-xs">From GitHub</Label>
        <Input id={ids.github} value={github} onChange={(e) => setGithub(e.target.value)} placeholder="octocat or a profile link" />
        <Button type="submit" disabled={!github.trim() || fromGithub.isPending} className="self-start">Read my GitHub</Button>
      </form>
      {error && (
        <p role="alert" className="text-danger sm:col-span-2">
          {error instanceof ApiError ? error.message : "Could not start the import."}
        </p>
      )}
    </div>
  );
}

export function SkillsStation() {
  const session = useCurrentSession();
  const { latest } = useLatestRun(session.id, SKILLS.kinds);
  return (
    <>
      <StationHeader station={SKILLS} title="What you can do today">
        Everything else on the route compares the market against this list. Imports add to it; edit it any time.
      </StationHeader>
      <SkillChips sessionId={session.id} skills={session.skills} />
      <ImportSkills sessionId={session.id} />
      <RunPanel sessionId={session.id} station={SKILLS} runId={latest?.id} />
    </>
  );
}
