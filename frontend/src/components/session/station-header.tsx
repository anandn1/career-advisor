import type { ReactNode } from "react";

import { StationRoundel } from "@/components/transit/station-roundel";
import type { Station } from "@/lib/stations";

export function StationHeader({ station, title, children }: { station: Station; title: string; children?: ReactNode }) {
  return (
    <div className="flex items-start gap-4 border-l-[6px] pl-4" style={{ borderColor: station.colour }}>
      <StationRoundel station={station} size="lg" />
      <div className="flex flex-col gap-1">
        <p className="signage text-xs text-ink-muted">{station.name}</p>
        <h1 className="font-heading text-3xl font-extrabold leading-tight sm:text-4xl">{title}</h1>
        {children && <div className="max-w-2xl text-ink-muted">{children}</div>}
      </div>
    </div>
  );
}
