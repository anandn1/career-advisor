import { Suspense } from "react";

import { SessionShell } from "@/components/session/session-shell";

export default function SessionLayout({ children }: LayoutProps<"/s/[sessionId]">) {
  // The session id is runtime data, so everything that reads it renders inside Suspense.
  return (
    <Suspense fallback={<p className="p-8 text-ink-muted">Loading your route…</p>}>
      <SessionShell>{children}</SessionShell>
    </Suspense>
  );
}
