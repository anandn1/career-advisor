import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { keys, useStartRun, useUpdateSession } from "@/hooks/use-career";
import type { Run } from "@/lib/api/client";
import { jsonResponse } from "@/test/render";

function makeRun(status: Run["status"]): Run {
  return { id: "r1", kind: "job_market", status, error: null, result: null, created_at: "", started_at: null, finished_at: null, events: [] };
}

function hookWith<T>(useHook: () => T) {
  const client = new QueryClient({ defaultOptions: { mutations: { retry: false } } });
  const invalidate = vi.spyOn(client, "invalidateQueries");
  const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  const { result } = renderHook(useHook, { wrapper });
  const invalidated = () => invalidate.mock.calls.map(([filters]) => filters?.queryKey);
  return { client, result, invalidated };
}

afterEach(() => vi.unstubAllGlobals());

describe("useStartRun", () => {
  it("seeds the new run's cache and refreshes the run list", async () => {
    const { client, result, invalidated } = hookWith(() => useStartRun("s1", async () => makeRun("queued")));
    await act(() => result.current.mutateAsync(undefined));
    expect(client.getQueryData(keys.run("s1", "r1"))).toEqual(makeRun("queued"));
    expect(invalidated()).toEqual([["runs", "s1"]]);
  });

  it("also refreshes the session and report when the run finished at once", async () => {
    const { result, invalidated } = hookWith(() => useStartRun("s1", async () => makeRun("succeeded")));
    await act(() => result.current.mutateAsync(undefined));
    expect(invalidated()).toEqual(expect.arrayContaining([keys.session("s1"), keys.report("s1")]));
  });

  it("caches nothing when starting fails", async () => {
    const { client, result } = hookWith(() => useStartRun("s1", async () => { throw new Error("busy"); }));
    await act(async () => { await expect(result.current.mutateAsync(undefined)).rejects.toThrow("busy"); });
    expect(client.getQueryData(keys.run("s1", "r1"))).toBeUndefined();
  });
});

describe("useUpdateSession", () => {
  it("stores the server's session and refreshes the report", async () => {
    const saved = { id: "s1", created_at: "", target_role: "Dev", location: "Pune", github_username: null, skills: ["Go", "Rust"] };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(saved)));
    const { client, result, invalidated } = hookWith(() => useUpdateSession("s1"));
    await act(() => result.current.mutateAsync({ skills: ["Go", "rust"] }));
    expect(client.getQueryData(keys.session("s1"))).toEqual(saved);
    expect(invalidated()).toEqual([keys.report("s1")]);
  });
});
