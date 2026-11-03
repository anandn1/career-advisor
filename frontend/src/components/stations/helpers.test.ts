import { describe, expect, it } from "vitest";

import type { Report } from "@/hooks/use-career";
import type { Interview } from "@/lib/api/client";
import { linesForTurn, parseTopics } from "@/components/stations/interview-station";
import { rankGaps } from "@/components/stations/market-station";
import { groupBySkill } from "@/components/stations/path-station";
import { stationSummaries } from "@/components/stations/report-station";
import { linesOf, skillsToText, textToSkills } from "@/components/stations/resume-station";

describe("rankGaps", () => {
  it("ranks gaps by how many postings ask for them, then by name", () => {
    const result = {
      jobs: [],
      analysis: {
        skill_gaps: ["aws", "docker", "go"],
        gaps_per_job: [
          { title: "A", company: "X", gaps: ["docker", "aws"], gap_count: 2 },
          { title: "B", company: "Y", gaps: ["docker"], gap_count: 1 },
        ],
      },
    };
    expect(rankGaps(result)).toEqual([
      { skill: "docker", postings: 2 },
      { skill: "aws", postings: 1 },
      { skill: "go", postings: 0 },
    ]);
  });

  it("breaks ties between equally common gaps alphabetically", () => {
    const result = {
      jobs: [],
      analysis: { gaps_per_job: [{ title: "A", company: "X", gaps: ["kafka", "aws"], gap_count: 2 }] },
    };
    expect(rankGaps(result).map((g) => g.skill)).toEqual(["aws", "kafka"]);
  });

  it("counts a gap listed twice in one posting as one posting", () => {
    const result = { jobs: [], analysis: { gaps_per_job: [{ title: "A", company: "X", gaps: ["aws", "aws"], gap_count: 2 }] } };
    expect(rankGaps(result)).toEqual([{ skill: "aws", postings: 1 }]);
  });

  it("handles an analysis without per-job gaps", () => {
    expect(rankGaps({ jobs: [], analysis: {} })).toEqual([]);
  });
});

describe("groupBySkill", () => {
  it("groups courses by skill in first-seen order", () => {
    const course = (skill: string, title: string) => ({ skill_name: skill, course_title: title, course_url: "u", description: "d" });
    const groups = groupBySkill({ user_summary: "", skill_gaps: [], recommendations: [course("go", "A"), course("aws", "B"), course("go", "C")] });
    expect(groups.map(([skill, courses]) => [skill, courses.map((c) => c.course_title)])).toEqual([["go", ["A", "C"]], ["aws", ["B"]]]);
  });
});

describe("resume text helpers", () => {
  it("turns non-blank lines into bullets", () => {
    expect(linesOf("  Built X\n\n Led Y  \n")).toEqual(["Built X", "Led Y"]);
  });

  it("round-trips skill categories through text", () => {
    const skills = [{ category_name: "Languages", skills: ["Go", "Python"] }, { category_name: "Cloud", skills: ["AWS"] }];
    expect(textToSkills(skillsToText(skills))).toEqual(skills);
  });

  it("puts a line without a category under 'Skills'", () => {
    expect(textToSkills("Go, Rust")).toEqual([{ category_name: "Skills", skills: ["Go", "Rust"] }]);
  });

  it("keeps colons that are part of the skill list", () => {
    expect(textToSkills("Tools: C:\\path, Git")).toEqual([{ category_name: "Tools", skills: ["C:\\path", "Git"] }]);
  });

  it("drops a category with no skills after its colon", () => {
    expect(textToSkills("Tools:\nLang: Go")).toEqual([{ category_name: "Lang", skills: ["Go"] }]);
  });

  it("puts skills under 'Skills' when the category name is blank", () => {
    expect(textToSkills(" : Go")).toEqual([{ category_name: "Skills", skills: ["Go"] }]);
  });

  it("drops empty skills and blank lines", () => {
    expect(textToSkills("Lang: Go,, ,Python\n\n")).toEqual([{ category_name: "Lang", skills: ["Go", "Python"] }]);
  });

  it("writes nothing for no skills", () => {
    expect(skillsToText(undefined)).toBe("");
  });
});

describe("interview helpers", () => {
  const interview = (turn: Partial<NonNullable<Interview["turn"]>> | null): Interview => ({
    id: "i1", status: "running", company_focus: "generic", topics: ["System Design"], report: [],
    turn: turn && { question: null, feedback: null, finished: false, summary: null, question_count: 1, ...turn },
  });

  it("shows feedback before the next question", () => {
    expect(linesForTurn(interview({ feedback: "Grade: GOOD", question: "Next?" }))).toEqual([
      { speaker: "feedback", text: "Grade: GOOD" },
      { speaker: "interviewer", text: "Next?" },
    ]);
  });

  it("adds nothing for a turn without feedback or question", () => {
    expect(linesForTurn(interview(null))).toEqual([]);
  });

  it("splits topics on commas and drops blanks", () => {
    expect(parseTopics(" Databases, , System Design ")).toEqual(["Databases", "System Design"]);
  });

  it("returns null when no topics are given, so the server picks them", () => {
    expect(parseTopics(" , ")).toBeNull();
  });
});

describe("stationSummaries", () => {
  const empty: Report = {
    session: { id: "s", created_at: "", target_role: "Dev", location: "Pune", github_username: null, skills: [] },
    job_market: null, learning_path: null, resume: null, resume_review: null, interviews: [],
  };

  it("reports nothing for an untouched route", () => {
    expect(Object.values(stationSummaries(empty)).every((line) => line === null)).toBe(true);
  });

  it("summarises each station that has results", () => {
    const report: Report = {
      ...empty,
      session: { ...empty.session, skills: ["Go"] },
      job_market: { run_id: "m", jobs: [], analysis: { skill_gaps: ["aws"] } },
      learning_path: { run_id: "p", path: { user_summary: "", skill_gaps: ["aws"], recommendations: [] } },
      resume: { run_id: "r", draft: {} as never, template_id: null, has_pdf: true, pdf_error: null },
      interviews: [{ id: "i", status: "succeeded", summary: "ok" }, { id: "j", status: "running", summary: null }],
    };
    expect(stationSummaries(report)).toEqual({
      skills: "1 skill: Go",
      market: "0 postings read; top gaps: aws",
      path: "0 courses across 1 skill",
      resume: "Resume written, PDF ready",
      interview: "1 mock interview finished",
    });
  });

  it("uses plurals for counts other than one", () => {
    const course = { skill_name: "go", course_title: "A", course_url: "u", description: "d" };
    const job = { title: "T", company: "C", location: "L", skills_required: [] };
    const report: Report = {
      ...empty,
      session: { ...empty.session, skills: ["Go", "SQL"] },
      job_market: { run_id: "m", jobs: [job, job], analysis: {} },
      learning_path: { run_id: "p", path: { user_summary: "", skill_gaps: ["go", "aws"], recommendations: [course, course] } },
      interviews: [{ id: "i", status: "succeeded", summary: "" }, { id: "j", status: "succeeded", summary: "" }],
    };
    const summaries = stationSummaries(report);
    expect([summaries.skills, summaries.market, summaries.path, summaries.interview]).toEqual([
      "2 skills: Go, SQL",
      "2 postings read; top gaps: none",
      "2 courses across 2 skills",
      "2 mock interviews finished",
    ]);
  });

  it("lists at most eight skills and marks the rest as cut off", () => {
    const skills = ["a", "b", "c", "d", "e", "f", "g", "h", "i"];
    expect(stationSummaries({ ...empty, session: { ...empty.session, skills } }).skills)
      .toBe("9 skills: a, b, c, d, e, f, g, h…");
  });

  it("lists exactly eight skills without a cut-off mark", () => {
    const skills = ["a", "b", "c", "d", "e", "f", "g", "h"];
    expect(stationSummaries({ ...empty, session: { ...empty.session, skills } }).skills)
      .toBe("8 skills: a, b, c, d, e, f, g, h");
  });

  it("names at most five gaps", () => {
    const analysis = { skill_gaps: ["a", "b", "c", "d", "e", "f"] };
    expect(stationSummaries({ ...empty, job_market: { run_id: "m", jobs: [], analysis } }).market)
      .toBe("0 postings read; top gaps: a, b, c, d, e");
  });

  it("says when the resume has no PDF", () => {
    const report: Report = { ...empty, resume: { run_id: "r", draft: {} as never, template_id: null, has_pdf: false, pdf_error: "x" } };
    expect(stationSummaries(report).resume).toBe("Resume written (.tex)");
  });

  it("prefers a written resume over a review", () => {
    const report: Report = {
      ...empty,
      resume: { run_id: "r", draft: {} as never, template_id: null, has_pdf: true, pdf_error: null },
      resume_review: { run_id: "v", review: { summary: "", strengths: [], weaknesses: [], suggestions: [] } },
    };
    expect(stationSummaries(report).resume).toBe("Resume written, PDF ready");
  });

  it("does not count unfinished interviews", () => {
    const report: Report = { ...empty, interviews: [{ id: "i", status: "running", summary: null }] };
    expect(stationSummaries(report).interview).toBeNull();
  });

  it("falls back to the review when no resume was written", () => {
    const report: Report = { ...empty, resume_review: { run_id: "v", review: { summary: "", strengths: [], weaknesses: [], suggestions: [] } } };
    expect(stationSummaries(report).resume).toBe("Existing resume reviewed");
  });
});
