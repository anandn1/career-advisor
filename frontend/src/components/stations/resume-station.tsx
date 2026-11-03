"use client";

import { Plus, Trash2 } from "lucide-react";
import { useId, useState } from "react";

import { RunPanel } from "@/components/session/run-panel";
import { useCurrentSession } from "@/components/session/session-shell";
import { StationHeader } from "@/components/session/station-header";
import { useLatestRun } from "@/components/stations/use-latest-run";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { type ResumeResult, type ResumeReview, useStartRun, useTemplates } from "@/hooks/use-career";
import { api, type ResumeDraft, runUrl, unwrap, uploadResume } from "@/lib/api/client";
import { isFinished, station } from "@/lib/stations";

const RESUME = station("resume");

type Education = { institution: string; degree: string; field: string; status: string; location: string; gpa: string };
type Work = { company: string; position: string; duration: string; location: string; bullets: string };

const emptyEducation = (): Education => ({ institution: "", degree: "", field: "", status: "", location: "", gpa: "" });
const emptyWork = (): Work => ({ company: "", position: "", duration: "", location: "", bullets: "" });

/** One bullet per non-blank line. */
export function linesOf(text: string): string[] {
  return text.split("\n").map((line) => line.trim()).filter(Boolean);
}

/** "Category: a, b" per line <-> skill categories. */
export function skillsToText(skills: ResumeDraft["skills"] = []): string {
  return skills.map((group) => `${group.category_name}: ${group.skills.join(", ")}`).join("\n");
}

export function textToSkills(text: string): NonNullable<ResumeDraft["skills"]> {
  return linesOf(text)
    .map((line) => {
      const [name, ...rest] = line.split(":");
      const list = rest.length ? rest.join(":") : name;
      return {
        category_name: (rest.length && name.trim()) || "Skills",
        skills: list.split(",").map((skill) => skill.trim()).filter(Boolean),
      };
    })
    .filter((category) => category.skills.length > 0);
}

function Field({ label, children }: { label: string; children: (id: string) => React.ReactNode }) {
  const id = useId();
  return (
    <div className="flex flex-col gap-1.5">
      <Label htmlFor={id} className="signage text-xs">{label}</Label>
      {children(id)}
    </div>
  );
}

function RepeatList<T>({ title, items, setItems, make, render }: {
  title: string; items: T[]; setItems: (items: T[]) => void; make: () => T;
  render: (item: T, update: (patch: Partial<T>) => void) => React.ReactNode;
}) {
  return (
    <fieldset className="flex flex-col gap-3">
      <legend className="signage text-xs">{title}</legend>
      {items.map((item, index) => (
        <div key={index} className="relative grid gap-3 rounded-lg border-2 border-rule p-4 sm:grid-cols-2">
          {render(item, (patch) => setItems(items.map((it, i) => (i === index ? { ...it, ...patch } : it))))}
          <Button type="button" variant="ghost" size="sm" className="justify-self-start text-ink-muted"
                  onClick={() => setItems(items.filter((_, i) => i !== index))}>
            <Trash2 className="size-4" /> Remove
          </Button>
        </div>
      ))}
      <Button type="button" variant="outline" size="sm" className="self-start" onClick={() => setItems([...items, make()])}>
        <Plus className="size-4" /> Add {title.toLowerCase().replace(/s$/, "")}
      </Button>
    </fieldset>
  );
}

function GenerateForm({ sessionId, disabled }: { sessionId: string; disabled: boolean }) {
  const session = useCurrentSession();
  const templates = useTemplates();
  const [contact, setContact] = useState({ name: "", email: "", phone: "", location: session.location, github: "" });
  const [education, setEducation] = useState<Education[]>([]);
  const [work, setWork] = useState<Work[]>([]);
  const [templateId, setTemplateId] = useState("");
  const [includeGithub, setIncludeGithub] = useState(true);
  const start = useStartRun(sessionId, async () =>
    unwrap(await api.POST("/api/v1/sessions/{session_id}/runs/resume", {
      params: { path: { session_id: sessionId } },
      body: {
        contact,
        education,
        work_experience: work.map(({ bullets, ...rest }) => ({ ...rest, description_bullets: linesOf(bullets) })),
        include_github: includeGithub,
        template_id: templateId || null,
      },
    })));

  return (
    <form className="flex flex-col gap-6 rounded-xl border-2 border-ink bg-surface p-6"
          onSubmit={(e) => { e.preventDefault(); start.mutate(undefined); }}>
      <div className="grid gap-4 sm:grid-cols-2">
        {(["name", "email", "phone", "location"] as const).map((key) => (
          <Field key={key} label={key === "name" ? "Full name" : key[0].toUpperCase() + key.slice(1)}>
            {(id) => (
              <Input id={id} value={contact[key]} type={key === "email" ? "email" : "text"}
                     onChange={(e) => setContact({ ...contact, [key]: e.target.value })} />
            )}
          </Field>
        ))}
      </div>
      <RepeatList title="Education" items={education} setItems={setEducation} make={emptyEducation}
        render={(item, update) => (
          <>
            <Field label="School">{(id) => <Input id={id} required value={item.institution} onChange={(e) => update({ institution: e.target.value })} />}</Field>
            <Field label="Degree">{(id) => <Input id={id} value={item.degree} onChange={(e) => update({ degree: e.target.value })} />}</Field>
            <Field label="Field">{(id) => <Input id={id} value={item.field} onChange={(e) => update({ field: e.target.value })} />}</Field>
            <Field label="Status">{(id) => <Input id={id} value={item.status} placeholder="Graduated May 2025" onChange={(e) => update({ status: e.target.value })} />}</Field>
            <Field label="Location">{(id) => <Input id={id} value={item.location} onChange={(e) => update({ location: e.target.value })} />}</Field>
            <Field label="Grade (optional)">{(id) => <Input id={id} value={item.gpa} placeholder="8.8/10" onChange={(e) => update({ gpa: e.target.value })} />}</Field>
          </>
        )} />
      <RepeatList title="Work experience" items={work} setItems={setWork} make={emptyWork}
        render={(item, update) => (
          <>
            <Field label="Company">{(id) => <Input id={id} required value={item.company} onChange={(e) => update({ company: e.target.value })} />}</Field>
            <Field label="Position">{(id) => <Input id={id} value={item.position} onChange={(e) => update({ position: e.target.value })} />}</Field>
            <Field label="Dates">{(id) => <Input id={id} value={item.duration} placeholder="Jan 2024 – Jun 2025" onChange={(e) => update({ duration: e.target.value })} />}</Field>
            <Field label="Location">{(id) => <Input id={id} value={item.location} onChange={(e) => update({ location: e.target.value })} />}</Field>
            <Field label="What you did, one per line">{(id) => <Textarea id={id} value={item.bullets} onChange={(e) => update({ bullets: e.target.value })} />}</Field>
          </>
        )} />
      <div className="grid gap-4 sm:grid-cols-2">
        <Field label="Template">
          {(id) => (
            <select id={id} value={templateId} onChange={(e) => setTemplateId(e.target.value)}
                    className="h-9 rounded-md border border-input bg-surface px-3">
              <option value="">Default template</option>
              {templates.data?.map((t) => (
                <option key={t.id} value={t.id}>{t.name}{t.tags.length ? ` (${t.tags.join(", ")})` : ""}</option>
              ))}
            </select>
          )}
        </Field>
        {session.github_username && (
          <label className="flex items-center gap-2 self-end">
            <input type="checkbox" checked={includeGithub} onChange={(e) => setIncludeGithub(e.target.checked)} />
            Add projects from GitHub ({session.github_username})
          </label>
        )}
      </div>
      {start.error && <p role="alert" className="text-danger">{start.error.message}</p>}
      <Button type="submit" size="lg" disabled={disabled || start.isPending} className="self-start font-heading font-bold">
        Write my resume
      </Button>
    </form>
  );
}

function DraftEditor({ sessionId, runId, result, disabled }: {
  sessionId: string; runId: string; result: ResumeResult; disabled: boolean;
}) {
  // The parent keys this editor by run id, so a new resume starts a fresh draft.
  const [summary, setSummary] = useState(result.draft.summary);
  const [skills, setSkills] = useState(skillsToText(result.draft.skills));
  const rerender = useStartRun(sessionId, async () =>
    unwrap(await api.POST("/api/v1/sessions/{session_id}/runs/resume-render", {
      params: { path: { session_id: sessionId } },
      body: { draft: { ...result.draft, summary, skills: textToSkills(skills) }, template_id: result.template_id },
    })));

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <form className="flex flex-col gap-4" onSubmit={(e) => { e.preventDefault(); rerender.mutate(undefined); }}>
        <Field label="Summary">{(id) => <Textarea id={id} rows={5} value={summary} onChange={(e) => setSummary(e.target.value)} />}</Field>
        <Field label="Skills, one category per line">
          {(id) => <Textarea id={id} rows={6} className="font-mono text-sm" value={skills} onChange={(e) => setSkills(e.target.value)} />}
        </Field>
        {rerender.error && <p role="alert" className="text-danger">{rerender.error.message}</p>}
        <div className="flex flex-wrap gap-3">
          <Button type="submit" disabled={disabled || rerender.isPending}>Re-typeset with my edits</Button>
          <Button asChild variant="outline"><a href={runUrl(sessionId, runId, "resume.tex")}>Download .tex</a></Button>
          {result.has_pdf && (
            <Button asChild variant="outline"><a href={`${runUrl(sessionId, runId, "resume.pdf")}?download=true`}>Download PDF</a></Button>
          )}
        </div>
      </form>
      {result.has_pdf ? (
        <iframe title="Resume preview" src={runUrl(sessionId, runId, "resume.pdf")}
                className="min-h-[36rem] w-full rounded-lg border-2 border-rule bg-surface" />
      ) : (
        <div className="rounded-lg border-2 border-dashed border-rule p-6 text-ink-muted">
          <p className="font-bold text-ink">No PDF preview</p>
          <p>{result.pdf_error ?? "The PDF could not be produced."} The .tex file opens in Overleaf or any LaTeX editor.</p>
        </div>
      )}
    </div>
  );
}

function ReviewResume({ sessionId }: { sessionId: string }) {
  const [file, setFile] = useState<File | null>(null);
  const [request, setRequest] = useState("");
  const { latest } = useLatestRun(sessionId, ["resume_review"]);
  const start = useStartRun(sessionId, () => uploadResume(sessionId, file!, "review", request));
  return (
    <section className="flex flex-col gap-4">
      <h2 className="font-heading text-2xl font-extrabold">Review a resume you already have</h2>
      <form className="grid gap-4 rounded-xl border-2 border-rule bg-surface p-5 sm:grid-cols-2"
            onSubmit={(e) => { e.preventDefault(); start.mutate(undefined); }}>
        <Field label="Resume file">{(id) => <Input id={id} type="file" accept=".pdf,.docx,.txt" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />}</Field>
        <Field label="Anything to focus on? (optional)">{(id) => <Input id={id} maxLength={1000} value={request} onChange={(e) => setRequest(e.target.value)} placeholder="Is it strong enough for senior roles?" />}</Field>
        {start.error && <p role="alert" className="text-danger sm:col-span-2">{start.error.message}</p>}
        <Button type="submit" disabled={!file || start.isPending || (latest !== undefined && !isFinished(latest))} className="justify-self-start">
          Review my resume
        </Button>
      </form>
      <RunPanel sessionId={sessionId} station={RESUME} runId={latest?.id}>
        {(run) => {
          const review = (run.result as unknown as { review: ResumeReview }).review;
          return (
            <div className="flex flex-col gap-4">
              <p className="text-lg">{review.summary}</p>
              <div className="grid gap-4 md:grid-cols-3">
                {([["Strengths", review.strengths], ["Weaknesses", review.weaknesses], ["Fix next", review.suggestions]] as const).map(
                  ([title, items]) => (
                    <div key={title} className="rounded-lg bg-surface p-4">
                      <h3 className="signage mb-2 text-xs">{title}</h3>
                      <ul className="list-disc space-y-1 pl-5">{items.map((item) => <li key={item}>{item}</li>)}</ul>
                    </div>
                  ),
                )}
              </div>
            </div>
          );
        }}
      </RunPanel>
    </section>
  );
}

export function ResumeStation() {
  const session = useCurrentSession();
  const { latest } = useLatestRun(session.id, ["resume", "resume_render"]);
  const busy = latest !== undefined && !isFinished(latest);
  return (
    <>
      <StationHeader station={RESUME} title={`A resume written for ${session.target_role} roles`}>
        One agent drafts the content from your details and GitHub; a second typesets it in LaTeX, following the template you pick.
      </StationHeader>
      <GenerateForm sessionId={session.id} disabled={busy} />
      <RunPanel sessionId={session.id} station={RESUME} runId={latest?.id}>
        {(run) => (
          <DraftEditor key={run.id} sessionId={session.id} runId={run.id} result={run.result as unknown as ResumeResult} disabled={busy} />
        )}
      </RunPanel>
      <ReviewResume sessionId={session.id} />
    </>
  );
}
