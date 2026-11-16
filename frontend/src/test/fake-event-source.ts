/** A stand-in for the browser's EventSource that tests drive by hand. */
export class FakeEventSource {
  static instances: FakeEventSource[] = [];
  readonly listeners = new Map<string, ((message: MessageEvent) => void)[]>();
  closed = false;

  constructor(readonly url: string) {
    FakeEventSource.instances.push(this);
  }

  addEventListener(type: string, listener: (message: MessageEvent) => void) {
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), listener]);
  }

  close() {
    this.closed = true;
  }

  /** Deliver a server-sent event with JSON data and an optional `id:`. */
  emit(type: string, data: unknown, lastEventId = "") {
    const message = new MessageEvent(type, { data: JSON.stringify(data), lastEventId });
    for (const listener of this.listeners.get(type) ?? []) listener(message);
  }
}
