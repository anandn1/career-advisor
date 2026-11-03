"use client";

import { useRuns } from "@/hooks/use-career";
import type { RunKind } from "@/lib/api/client";

/** The newest run of any of `kinds` (runs come back newest first). */
export function useLatestRun(sessionId: string, kinds: RunKind[]) {
  const runs = useRuns(sessionId, kinds);
  return { latest: runs.data?.[0], isPending: runs.isPending };
}
