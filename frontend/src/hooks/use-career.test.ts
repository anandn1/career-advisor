import { describe, expect, it } from "vitest";

import { placeEvent } from "@/hooks/use-career";

describe("placeEvent", () => {
  it("appends the next event", () => {
    expect(placeEvent(["a", "b"], 2, "c")).toEqual(["a", "b", "c"]);
  });

  it("replaces an event replayed at its position instead of duplicating it", () => {
    expect(placeEvent(["a", "b", "c"], 1, "b2")).toEqual(["a", "b2", "c"]);
  });

  it("keeps later events when an early one is replayed", () => {
    expect(placeEvent(["a", "b", "c"], 0, "a")).toEqual(["a", "b", "c"]);
  });

  it.each([[Number.NaN], [-1], [1.5], [9]])("appends when the position %s is unusable", (index) => {
    expect(placeEvent(["a"], index, "z")).toEqual(["a", "z"]);
  });

  it("does not change the original list", () => {
    const events = ["a"];
    placeEvent(events, 0, "b");
    expect(events).toEqual(["a"]);
  });
});
