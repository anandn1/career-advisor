import { cn } from "@/lib/utils";
import type { Station } from "@/lib/stations";

type Size = "sm" | "md" | "lg";

const SIZES: Record<Size, string> = {
  sm: "size-7 text-sm",
  md: "size-9 text-base",
  lg: "size-14 text-2xl",
};

/** A lettered station marker, like a transit line roundel. `filled` means the station has results. */
export function StationRoundel({
  station,
  size = "md",
  filled = true,
  className,
}: {
  station: Station;
  size?: Size;
  filled?: boolean;
  className?: string;
}) {
  return (
    <span
      aria-hidden
      className={cn(
        "inline-grid shrink-0 place-items-center rounded-full border-[3px] font-heading font-extrabold leading-none",
        SIZES[size],
        className,
      )}
      style={{
        borderColor: station.colour,
        background: filled ? station.colour : "var(--color-surface)",
        color: filled ? station.onColour : station.colour,
      }}
    >
      {station.letter}
    </span>
  );
}
