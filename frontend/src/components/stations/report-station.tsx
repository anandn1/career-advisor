"use client";

import { useMutation } from "@tanstack/react-query";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { useCurrentSession } from "@/components/session/session-shell";
import { rankGaps } from "@/components/stations/market-station";
import { StationRoundel } from "@/components/transit/station-roundel";
import { Button } from "@/components/ui/button";
import { forgetSession, type Report, useReport } from "@/hooks/use-career";
import { api, unwrap } from "@/lib/api/client";
import { STATIONS, type StationId } from "@/lib/stations";

/** One line per station: what the route has produced there so far (null if nothing yet). */
export function stationSummaries(report: Report): Record<StationId, string | null> {
  const skills = report.session.skills;
  const market = report.job_market;
  const path = report.learning_path?.path;
  const finishedInterviews = report.interviews.filter((i) => i.status === "succeeded").length;
  return {
    skills: skills.length ? `${skills.length} skills: ${skills.slice(0, 8).join(", ")}${skills.length > 8 ? "…" : ""}` : null,
    market: market
      ? `${market.jobs.length} postings read; top gaps: ${rankGaps(market).slice(0, 5).map((g) => g.skill).join(", ") || "none"}`
      : null,
    path: path ? `${path.recommendations.length} courses across ${path.skill_gaps.length} skills` : null,
    resume: report.resume ? (report.resume.has_pdf ? "Resume written, PDF ready" : "Resume written (.tex)") :
      report.resume_review ? "Existing resume reviewed" : null,
    interview: finishedInterviews ? `${finishedInterviews} mock interview${finishedInterviews > 1 ? "s" : ""} finished` : null,
  };
}

function DeleteRoute({ sessionId }: { sessionId: string }) {
  const router = useRouter();
  const [confirming, setConfirming] = useState(false);
  const remove = useMutation({
    mutationFn: async () =>
      unwrap(await api.DELETE("/api/v1/sessions/{session_id}", { params: { path: { session_id: sessionId } } })),
    onSuccess: () => { forgetSession(); router.push("/"); },
  });
  return (
    <section className="flex flex-col gap-3 border-t-2 border-rule pt-6">
      <h2 className="font-heading text-xl font-extrabold">Delete this route</h2>
      <p className="text-ink-muted">Removes your skills, results, resumes and interview transcripts now. Unused routes are deleted after 30 days anyway.</p>
      {remove.error && <p role="alert" className="text-danger">{remove.error.message}</p>}
      {confirming ? (
        <div className="flex gap-3">
          <Button variant="destructive" disabled={remove.isPending} onClick={() => remove.mutate()}>Delete everything</Button>
          <Button variant="outline" onClick={() => setConfirming(false)}>Keep it</Button>
        </div>
      ) : (
        <Button variant="outline" className="self-start" onClick={() => setConfirming(true)}>Delete my data</Button>
      )}
    </section>
  );
}

export function ReportStation() {
  const session = useCurrentSession();
  const report = useReport(session.id);
  if (!report.data) return <p className="text-ink-muted">Loading the report…</p>;
  const summaries = stationSummaries(report.data);
  return (
    <>
      <div className="flex flex-col gap-2">
        <p className="signage text-xs text-ink-muted">End of the line</p>
        <h1 className="font-heading text-4xl font-extrabold">Your route to {session.target_role}</h1>
      </div>
      <ol className="flex flex-col">
        {STATIONS.map((station) => (
          <li key={station.id} className="flex items-start gap-4 border-l-[6px] py-3 pl-4" style={{ borderColor: station.colour }}>
            <StationRoundel station={station} filled={summaries[station.id] !== null} />
            <div className="flex flex-col">
              <Link href={`/s/${session.id}/${station.id}`} className="font-heading text-lg font-extrabold hover:underline">{station.name}</Link>
              <span className={summaries[station.id] ? "" : "text-ink-muted"}>{summaries[station.id] ?? "Not visited yet"}</span>
            </div>
          </li>
        ))}
      </ol>
      <DeleteRoute sessionId={session.id} />
    </>
  );
}
