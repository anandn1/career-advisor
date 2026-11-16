import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";

import { expect, type Locator, type Page, test } from "@playwright/test";

/*
 * The demo: a fictional candidate, Asha Rao, goes through every station with real agents.
 * Writes one screenshot per station to docs/screenshots/ and, next to the video in
 * recording/output/, the times of each wait so `speed-up-waits.mjs` can speed them up.
 */

const HERE = __dirname; // Playwright loads this file as CommonJS
const OUTPUT = path.join(HERE, "output");
const SCREENSHOTS = path.join(HERE, "..", "..", "docs", "screenshots");
const RESUME_FILE = path.join(HERE, "fixtures", "asha-rao-resume.txt");
const RUN_TIMEOUT = 25 * 60_000;
const READING_PAUSE = 2_500; // let viewers read a result before moving on
const TYPING_DELAY = 35;

type Wait = { label: string; start: number; end: number };

test.use({ channel: "chromium" }); // full Chromium, which can show the PDF preview

test("Asha's route to Backend Engineer", async ({ page }) => {
  const startedAt = Date.now();
  const waits: Wait[] = [];
  const seconds = () => (Date.now() - startedAt) / 1000;
  mkdirSync(OUTPUT, { recursive: true });
  mkdirSync(SCREENSHOTS, { recursive: true });

  /** Wait for the station's agent run to finish, noting the wait for the video edit. */
  async function waitForRun(label: string) {
    const travelling = page.getByText("On the line now");
    // A quick run can finish before its progress is ever shown.
    await travelling.waitFor({ state: "visible", timeout: 15_000 }).catch(() => {});
    const start = seconds();
    await expect(travelling).toBeHidden({ timeout: RUN_TIMEOUT });
    waits.push({ label, start, end: seconds() });
    // Errors show as alerts in the page (Next.js also announces each page title as one).
    await expect(page.getByRole("main").getByRole("alert")).toHaveCount(0);
  }

  async function shot(name: string) {
    await page.waitForTimeout(READING_PAUSE);
    // A full-page capture would paint the sticky route bar where the page is scrolled to.
    const unstick = await page.addStyleTag({ content: "header.sticky { position: static !important; }" });
    await page.screenshot({ path: path.join(SCREENSHOTS, `${name}.png`), fullPage: true });
    await unstick.evaluate((style) => style.remove());
  }

  const type = (field: Locator, text: string) => field.pressSequentially(text, { delay: TYPING_DELAY });
  const station = (name: RegExp) =>
    page.getByRole("navigation", { name: "Route to your target role" }).getByRole("link", { name });

  try {
    // Start: role, place and a resume.
    await page.goto("/");
    await type(page.getByLabel("Target role"), "Backend Engineer");
    await type(page.getByLabel("Where"), "Bengaluru");
    await page.getByLabel("Upload a resume").check();
    await page.getByLabel(/PDF, DOCX or TXT/).setInputFiles(RESUME_FILE);
    await shot("01-start");
    await page.getByRole("button", { name: "Start the route" }).click();

    // Skills read from the resume.
    await waitForRun("Reading skills from the resume");
    await expect(page.getByRole("list", { name: "Your skills" })).toBeVisible();
    await shot("02-skills");

    // Job market: live LinkedIn postings and the skill gaps.
    await station(/Market/).click();
    await page.getByRole("button", { name: "Analyze the job market" }).click();
    await waitForRun("Scraping LinkedIn postings and comparing skills");
    await expect(page.getByRole("heading", { name: "Skills you are missing" })).toBeVisible();
    await shot("03-market");

    // Learning path for the gaps.
    await station(/Learning path/).click();
    await page.getByRole("button", { name: "Plan my courses" }).click();
    await waitForRun("Planning, searching and scoring courses, then a review");
    await shot("04-learning-path");

    // Resume written for the role, compiled to PDF.
    await station(/Resume/).click();
    await type(page.getByLabel("Full name"), "Asha Rao");
    await type(page.getByLabel("Email"), "asha.rao@example.com");
    await page.getByLabel("Location").first().fill("Bengaluru"); // prefilled from the route
    await page.getByRole("button", { name: "Add education" }).click();
    await type(page.getByLabel("School"), "Deccan Institute of Technology");
    await type(page.getByLabel("Degree"), "B.Tech");
    await type(page.getByLabel("Field"), "Computer Science");
    await type(page.getByLabel("Status"), "Graduated 2022");
    await page.getByRole("button", { name: "Add work experience" }).click();
    await type(page.getByLabel("Company"), "Acme Payments");
    await type(page.getByLabel("Position"), "Backend Engineer");
    await type(page.getByLabel("Dates"), "Jul 2022 – present");
    await type(
      page.getByLabel("What you did, one per line"),
      "Built FastAPI services handling 2 million requests a day\nMoved batch jobs to AWS Lambda with Terraform",
    );
    await page.getByRole("button", { name: "Write my resume" }).click();
    await waitForRun("Drafting the resume and typesetting it in LaTeX");
    await expect(page.getByTitle("Resume preview")).toBeVisible();
    await page.getByTitle("Resume preview").scrollIntoViewIfNeeded();
    await shot("05-resume");

    // Review of an existing resume.
    await page.getByLabel("Resume file").setInputFiles(RESUME_FILE);
    await type(page.getByLabel(/Anything to focus on/), "Is it ready for backend roles?");
    await page.getByRole("button", { name: "Review my resume" }).click();
    await waitForRun("Reviewing the resume");
    await shot("06-resume-review");

    // Mock interview on System Design: one real answer, one "I don't know", then finish.
    await station(/Interview/).click();
    await chooseOnly(page, "System Design");
    await page.getByRole("button", { name: "Start the interview" }).click();
    const transcript = page.getByRole("list", { name: "Interview transcript" });
    await expect(transcript.getByRole("listitem")).not.toHaveCount(0, { timeout: RUN_TIMEOUT });
    await answer(page, "I would start with the requirements and expected load, then put a cache in front of "
      + "the database, send slow work to a queue, and shard the data by user ID as it grows.");
    await answer(page, "I don't know.");
    await page.getByRole("button", { name: "Finish interview" }).click();
    await expect(page.getByRole("heading", { name: "How it went" })).toBeVisible({ timeout: RUN_TIMEOUT });
    await shot("07-interview");

    // The report gathers every station.
    await page.getByRole("navigation", { name: "Route to your target role" }).getByRole("link", { name: /^To/ }).click();
    await expect(page.getByRole("heading", { name: /Your route to/ })).toBeVisible();
    await shot("08-report");
  } finally {
    writeFileSync(path.join(OUTPUT, "waits.json"), JSON.stringify(waits, null, 2));
  }
});

/** Leave exactly one topic chosen in the interview's topic picker. */
async function chooseOnly(page: Page, topic: string) {
  const picker = page.getByRole("group", { name: "Topics" });
  await expect(picker.getByRole("button", { name: topic })).toBeVisible();
  // Names first: a pressed-buttons locator changes as each one is unpressed.
  const chosen = await picker.getByRole("button", { pressed: true }).allTextContents();
  for (const name of chosen.filter((name) => name !== topic)) {
    await picker.getByRole("button", { name, exact: true }).click();
  }
  const wanted = picker.getByRole("button", { name: topic });
  if ((await wanted.getAttribute("aria-pressed")) !== "true") await wanted.click();
}

/** Type an answer, send it, and wait for the next line from the interviewer. */
async function answer(page: Page, text: string) {
  const lines = page.getByRole("list", { name: "Interview transcript" }).getByRole("listitem");
  const before = await lines.count();
  await page.getByLabel("Your answer").pressSequentially(text, { delay: TYPING_DELAY });
  await page.getByRole("button", { name: "Send answer" }).click();
  await expect(lines).not.toHaveCount(before, { timeout: RUN_TIMEOUT });
  await expect(page.getByRole("button", { name: "Send answer" })).toBeVisible({ timeout: RUN_TIMEOUT });
}
