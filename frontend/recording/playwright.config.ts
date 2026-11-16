import { defineConfig } from "@playwright/test";

// Records the demo video: one real run of every agent against the running Docker stack.
// `pnpm record` (APP_URL to point elsewhere, e.g. http://localhost:3001).
export default defineConfig({
  testDir: ".",
  testMatch: "demo.spec.ts",
  outputDir: "output/playwright",
  timeout: 40 * 60_000, // the LinkedIn scrape alone takes about ten minutes
  workers: 1,
  reporter: "list",
  use: {
    baseURL: process.env.APP_URL ?? "http://localhost:3000",
    viewport: { width: 1280, height: 720 },
    video: { mode: "on", size: { width: 1280, height: 720 } },
    launchOptions: { slowMo: 60 },
    actionTimeout: 60_000, // a click that cannot happen fails instead of hanging
  },
});
