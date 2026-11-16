"use client";

import Link from "next/link";

import { StationRoundel } from "@/components/transit/station-roundel";
import { cn } from "@/lib/utils";
import { STATIONS, type StationId } from "@/lib/stations";

export type StationStatus = "open" | "running" | "done";

/** On narrow screens the bar scrolls; keep the station being viewed on screen. */
function scrollIntoViewOnMount(element: HTMLAnchorElement | null) {
  element?.scrollIntoView?.({ block: "nearest", inline: "center" });
}

/**
 * The route from the user's skills to the target role. Each segment of line takes the colour
 * of the station it leads to; a station is filled once it has results.
 */
export function RouteBar({
  sessionId,
  current,
  status,
  terminus,
}: {
  sessionId: string;
  current: StationId | "report";
  status: Partial<Record<StationId, StationStatus>>;
  terminus: string;
}) {
  return (
    <nav aria-label="Route to your target role" className="relative w-full min-w-0 overflow-x-auto">
      <ol className="flex w-full min-w-max items-center gap-0 py-3">
        <li className="signage pr-3 text-xs text-ink-muted">You</li>
        {STATIONS.map((station, index) => {
          const state = status[station.id] ?? "open";
          const isCurrent = current === station.id;
          return (
            <li key={station.id} className="flex flex-1 items-center">
              <span
                aria-hidden
                className="h-[5px] min-w-6 flex-1 rounded-full"
                style={{
                  background: station.colour,
                  opacity: index === 0 || state !== "open" ? 1 : 0.35,
                }}
              />
              <Link
                href={`/s/${sessionId}/${station.id}`}
                aria-current={isCurrent ? "page" : undefined}
                ref={isCurrent ? scrollIntoViewOnMount : undefined}
                className={cn(
                  "group flex items-center gap-2 rounded-full px-1.5 py-1 transition-colors hover:bg-surface",
                  isCurrent && "bg-surface shadow-[0_0_0_2px_var(--color-ink)]",
                )}
              >
                <span
                  className={cn(
                    "rounded-full",
                    state === "running" && "animate-[stop-pulse_1.6s_ease-in-out_infinite]",
                  )}
                  style={{ "--stop-colour": station.colour } as React.CSSProperties}
                >
                  <StationRoundel station={station} size="sm" filled={state === "done"} />
                </span>
                <span className="signage text-xs whitespace-nowrap text-ink">{station.name}</span>
                <span className="sr-only">
                  {state === "done" ? "(has results)" : state === "running" ? "(running)" : "(not started)"}
                </span>
              </Link>
            </li>
          );
        })}
        <li className="flex flex-1 items-center">
          <span aria-hidden className="h-[5px] min-w-6 flex-1 rounded-full bg-ink" />
          <Link
            href={`/s/${sessionId}/report`}
            aria-current={current === "report" ? "page" : undefined}
            className={cn(
              "flex items-center gap-2 rounded-md border-2 border-ink bg-ink px-2.5 py-1 text-surface",
              current === "report" && "shadow-[0_0_0_3px_var(--color-line-path)]",
            )}
          >
            <span className="signage text-[0.65rem] text-surface/70">To</span>
            <span className="max-w-64 truncate font-heading text-sm font-extrabold">{terminus}</span>
          </Link>
        </li>
      </ol>
    </nav>
  );
}
