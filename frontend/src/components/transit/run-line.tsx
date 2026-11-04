import { cn } from "@/lib/utils";
import { stopProgress, type Station } from "@/lib/stations";
import type { Run } from "@/lib/api/client";

/** A run drawn as a line of stops: the agent's real steps light up as it passes them. */
export function RunLine({ run, station }: { run: Run; station: Station }) {
  const stops = stopProgress(run);
  const failed = run.status === "failed";
  return (
    <ol aria-label={`${station.name} progress`} className="relative ml-3 border-l-[5px] py-1"
        style={{ borderColor: `color-mix(in srgb, ${station.colour} 30%, transparent)` }}>
      {stops.map(({ stop, state, visits }) => (
        <li key={stop.id} className="relative flex items-baseline gap-3 py-2 pl-6">
          <span
            aria-hidden
            className={cn(
              "absolute top-3 -left-[13px] size-[21px] rounded-full border-[4px] bg-surface",
              state === "current" && "animate-[stop-pulse_1.6s_ease-in-out_infinite]",
            )}
            style={
              {
                borderColor: state === "waiting" ? "var(--color-rule)" : station.colour,
                background: state === "passed" ? station.colour : "var(--color-surface)",
                "--stop-colour": station.colour,
              } as React.CSSProperties
            }
          />
          <span className={cn("font-heading font-bold", state === "waiting" ? "text-ink-muted" : "text-ink")}>
            {stop.label}
          </span>
          {visits > 1 && <span className="font-mono text-xs text-ink-muted">×{visits}</span>}
          <span className="sr-only">
            {state === "passed" ? "done" : state === "current" ? "in progress" : "not reached"}
          </span>
        </li>
      ))}
      <li className="relative flex items-baseline gap-3 py-2 pl-6">
        <span
          aria-hidden
          className="absolute top-2.5 -left-[15px] h-[25px] w-[25px] rounded-sm border-[4px] bg-surface"
          style={{
            borderColor: failed ? "var(--color-danger)" : run.status === "succeeded" ? station.colour : "var(--color-rule)",
            background: run.status === "succeeded" ? station.colour : "var(--color-surface)",
          }}
        />
        <span className={cn("font-heading font-bold",
                            failed ? "text-danger" : run.status === "succeeded" ? "text-ink" : "text-ink-muted")}>
          {failed ? "Stopped" : run.status === "succeeded" ? "Arrived" : "Arrival"}
        </span>
      </li>
    </ol>
  );
}
