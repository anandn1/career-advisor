import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { keys, useRunStream } from "@/hooks/use-career";
import type { Run } from "@/lib/api/client";
import { FakeEventSource } from "@/test/fake-event-source";

const event = (step: string) => ({ step, detail: {}, at: "" });

function makeRun(status: Run["status"], steps: string[] = []): Run {
  return {
    id: "r1", kind: "job_market", status, error: null, result: null, created_at: "", started_at: null,
    finished_at: null, events: steps.map(event),
  };
}

function setup(run: Run | undefined) {
  const client = new QueryClient();
  if (run) client.setQueryData(keys.run("s1", run.id), run);
  const invalidate = vi.spyOn(client, "invalidateQueries");
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
  const hook = renderHook(({ current }: { current: Run | undefined }) => useRunStream("s1", current), {
    wrapper, initialProps: { current: run },
  });
  const cached = () => client.getQueryData<Run>(keys.run("s1", "r1"));
  return { client, hook, invalidate, cached, source: () => FakeEventSource.instances.at(-1)! };
}

beforeEach(() => {
  FakeEventSource.instances = [];
  vi.stubGlobal("EventSource", FakeEventSource);
});
afterEach(() => vi.unstubAllGlobals());

describe("useRunStream", () => {
  it("opens the run's event stream while it is unfinished", () => {
    const { source } = setup(makeRun("queued"));
    expect(FakeEventSource.instances).toHaveLength(1);
    expect(source().url).toMatch(/\/api\/v1\/sessions\/s1\/runs\/r1\/events$/);
  });

  it.each(["succeeded", "failed"] as const)("opens no stream for a %s run", (status) => {
    setup(makeRun(status));
    expect(FakeEventSource.instances).toHaveLength(0);
  });

  it("opens no stream without a run", () => {
    setup(undefined);
    expect(FakeEventSource.instances).toHaveLength(0);
  });

  it("adds progress events to the cached run and marks it running", () => {
    const { source, cached } = setup(makeRun("queued"));
    act(() => source().emit("progress", event("search_cached_postings"), "0"));
    expect(cached()?.status).toBe("running");
    expect(cached()?.events.map((e) => e.step)).toEqual(["search_cached_postings"]);
  });

  it("does not duplicate events replayed after a reconnect", () => {
    const { source, cached } = setup(makeRun("running", ["a", "b"]));
    act(() => {
      source().emit("progress", event("a"), "0");
      source().emit("progress", event("b"), "1");
      source().emit("progress", event("c"), "2");
    });
    expect(cached()?.events.map((e) => e.step)).toEqual(["a", "b", "c"]);
  });

  it("appends an event that arrives without an id", () => {
    const { source, cached } = setup(makeRun("running", ["a", "b"]));
    act(() => source().emit("progress", event("c")));
    expect(cached()?.events.map((e) => e.step)).toEqual(["a", "b", "c"]);
  });

  it("leaves the cache alone when the run is not cached", () => {
    const run = makeRun("running");
    const client = new QueryClient();
    const wrapper = ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
    renderHook(() => useRunStream("s1", run), { wrapper });
    act(() => FakeEventSource.instances[0].emit("progress", event("a"), "0"));
    expect(client.getQueryData(keys.run("s1", "r1"))).toBeUndefined();
  });

  it("stores the finished run, closes the stream and refreshes what the run changed", () => {
    const { source, cached, invalidate } = setup(makeRun("running"));
    const finished = makeRun("succeeded", ["analyze_skill_gaps"]);
    act(() => source().emit("done", finished));
    expect(cached()).toEqual(finished);
    expect(source().closed).toBe(true);
    const refreshed = invalidate.mock.calls.map(([filters]) => filters?.queryKey);
    expect(refreshed).toHaveLength(3);
    expect(refreshed).toEqual(expect.arrayContaining([keys.session("s1"), keys.report("s1"), ["runs", "s1"]]));
  });

  it("closes the stream on unmount", () => {
    const { source, hook } = setup(makeRun("running"));
    hook.unmount();
    expect(source().closed).toBe(true);
  });

  it("closes the stream once the run it follows is finished", () => {
    const { source, hook } = setup(makeRun("running"));
    hook.rerender({ current: makeRun("succeeded") });
    expect(source().closed).toBe(true);
    expect(FakeEventSource.instances).toHaveLength(1);
  });
});
