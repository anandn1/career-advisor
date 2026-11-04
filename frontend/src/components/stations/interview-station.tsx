"use client";

import { useMutation } from "@tanstack/react-query";
import { useId, useState } from "react";

import { useCurrentSession } from "@/components/session/session-shell";
import { StationHeader } from "@/components/session/station-header";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { useInterview, useReport } from "@/hooks/use-career";
import { api, type Interview, unwrap } from "@/lib/api/client";
import { station } from "@/lib/stations";

const INTERVIEW = station("interview");
const COMPANIES = ["generic", "amazon", "apple", "google", "meta", "microsoft", "netflix", "nvidia", "all"] as const;
type Company = (typeof COMPANIES)[number];

type Line = { speaker: "interviewer" | "you" | "feedback"; text: string };

/** Transcript lines added by one turn of the interview. */
export function linesForTurn(interview: Interview): Line[] {
  const lines: Line[] = [];
  if (interview.turn?.feedback) lines.push({ speaker: "feedback", text: interview.turn.feedback });
  if (interview.turn?.question) lines.push({ speaker: "interviewer", text: interview.turn.question });
  return lines;
}

export function parseTopics(text: string): string[] | null {
  const topics = text.split(",").map((topic) => topic.trim()).filter(Boolean);
  return topics.length ? topics : null;
}

function Transcript({ lines }: { lines: Line[] }) {
  return (
    <ol className="flex flex-col gap-3" aria-label="Interview transcript">
      {lines.map((line, index) => (
        <li key={index} className={line.speaker === "you" ? "self-end max-w-[85%]" : "max-w-[85%]"}>
          {line.speaker === "interviewer" && (
            <div className="rounded-xl border-l-[6px] bg-surface p-4" style={{ borderColor: INTERVIEW.colour }}>
              <p className="signage mb-1 text-[0.65rem] text-ink-muted">Interviewer</p>
              <p className="whitespace-pre-wrap text-lg">{line.text}</p>
            </div>
          )}
          {line.speaker === "you" && <p className="rounded-xl bg-ink p-4 whitespace-pre-wrap text-surface">{line.text}</p>}
          {line.speaker === "feedback" && (
            <p className="rounded-lg border-2 border-dashed border-rule p-3 font-mono text-sm whitespace-pre-wrap text-ink-muted">{line.text}</p>
          )}
        </li>
      ))}
    </ol>
  );
}

function Results({ sessionId, interviewId }: { sessionId: string; interviewId: string }) {
  const { data } = useInterview(sessionId, interviewId);
  const entries = data?.report ?? [];
  if (entries.length === 0) return null;
  return (
    <table className="w-full border-collapse text-left text-sm">
      <thead><tr className="signage text-xs text-ink-muted"><th className="py-2">Question</th><th>Topic</th><th>Grade</th></tr></thead>
      <tbody>
        {entries.map((entry) => (
          <tr key={String(entry.question_id)} className="border-t-2 border-rule align-top">
            <td className="py-2 pr-4">{String(entry.question)}</td>
            <td className="pr-4">{String(entry.topic)}</td>
            <td className="font-mono">{String(entry.evaluation)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function InterviewStation() {
  const session = useCurrentSession();
  const report = useReport(session.id);
  const ids = { company: useId(), topics: useId(), answer: useId() };
  const [company, setCompany] = useState<Company>("generic");
  const [topics, setTopics] = useState("");
  const [interview, setInterview] = useState<Interview | null>(null);
  const [lines, setLines] = useState<Line[]>([]);
  const [answer, setAnswer] = useState("");
  const sessionPath = { session_id: session.id };

  const start = useMutation({
    mutationFn: async () =>
      unwrap(await api.POST("/api/v1/sessions/{session_id}/interviews", {
        params: { path: sessionPath }, body: { company_focus: company, topics: parseTopics(topics) },
      })),
    onSuccess: (started) => { setInterview(started); setLines(linesForTurn(started)); },
  });
  const reply = useMutation({
    mutationFn: async (text: string) =>
      unwrap(await api.POST("/api/v1/sessions/{session_id}/interviews/{interview_id}/answers", {
        params: { path: { ...sessionPath, interview_id: interview!.id } }, body: { answer: text },
      })),
    onSuccess: (next, text) => {
      setInterview(next);
      setLines((current) => [...current, { speaker: "you", text }, ...linesForTurn(next)]);
      setAnswer("");
    },
  });
  const finished = interview?.turn?.finished ?? false;

  return (
    <>
      <StationHeader station={INTERVIEW} title="Practice the interview">
        Questions come from real company question banks, aimed at your gaps. Strong answers make the next one
        harder; weak ones may get a follow-up.
      </StationHeader>

      {!interview && (
        <form className="grid gap-4 rounded-xl border-2 border-ink bg-surface p-6 sm:grid-cols-2"
              onSubmit={(e) => { e.preventDefault(); start.mutate(); }}>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor={ids.company} className="signage text-xs">Company questions</Label>
            <select id={ids.company} value={company} onChange={(e) => setCompany(e.target.value as Company)}
                    className="h-9 rounded-md border border-input bg-surface px-3">
              {COMPANIES.map((name) => <option key={name} value={name}>{name === "all" ? "All companies" : name[0].toUpperCase() + name.slice(1)}</option>)}
            </select>
          </div>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor={ids.topics} className="signage text-xs">Topics (optional, comma separated)</Label>
            <Input id={ids.topics} value={topics} onChange={(e) => setTopics(e.target.value)} placeholder="Based on your skill gaps" />
          </div>
          {start.error && <p role="alert" className="text-danger sm:col-span-2">{start.error.message}</p>}
          <Button type="submit" size="lg" disabled={start.isPending} className="self-start font-heading font-bold">
            {start.isPending ? "Picking the first question…" : "Start the interview"}
          </Button>
        </form>
      )}

      {interview && (
        <section className="flex flex-col gap-5">
          <p className="signage text-xs text-ink-muted">
            {interview.topics.join(" · ")} · question {interview.turn?.question_count ?? 0}
          </p>
          <Transcript lines={lines} />
          {!finished ? (
            <form className="flex flex-col gap-3" onSubmit={(e) => { e.preventDefault(); reply.mutate(answer); }}>
              <Label htmlFor={ids.answer} className="signage text-xs">Your answer</Label>
              <Textarea id={ids.answer} rows={5} maxLength={5000} value={answer} onChange={(e) => setAnswer(e.target.value)} />
              {reply.error && <p role="alert" className="text-danger">{reply.error.message}</p>}
              <div className="flex flex-wrap gap-3">
                <Button type="submit" disabled={reply.isPending || !answer.trim()}>
                  {reply.isPending ? "Grading…" : "Send answer"}
                </Button>
                <Button type="button" variant="outline" disabled={reply.isPending} onClick={() => reply.mutate("stop")}>
                  Finish interview
                </Button>
              </div>
            </form>
          ) : (
            <div className="flex flex-col gap-4 rounded-xl border-2 border-ink bg-surface p-6">
              <h2 className="font-heading text-2xl font-extrabold">How it went</h2>
              <p className="whitespace-pre-wrap">{interview.turn?.summary?.replace(/^Interview complete!\s*/, "")}</p>
              <Results sessionId={session.id} interviewId={interview.id} />
              <Button className="self-start" onClick={() => { setInterview(null); setLines([]); }}>Start another interview</Button>
            </div>
          )}
        </section>
      )}

      {!interview && (report.data?.interviews.length ?? 0) > 0 && (
        <section className="flex flex-col gap-2">
          <h2 className="signage text-sm">Earlier interviews</h2>
          <ul className="flex flex-col gap-2">
            {report.data!.interviews.map((past) => (
              <li key={past.id} className="rounded-lg bg-surface p-3 text-sm">
                <span className="signage mr-2 text-[0.65rem] text-ink-muted">{past.status}</span>
                {past.summary?.replace(/^Interview complete!\s*/, "").slice(0, 220) ?? "In progress"}
              </li>
            ))}
          </ul>
        </section>
      )}
    </>
  );
}
