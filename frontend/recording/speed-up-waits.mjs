// Turns the raw demo recording into recording/output/career-advisor-demo.mp4: long agent
// waits (from waits.json) play in a few seconds with a label saying how long they took.
import { execFileSync } from "node:child_process";
import { mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const OUTPUT = path.join(HERE, "output");
const LABELS = path.join(OUTPUT, "labels");
const SPED_UP_SECONDS = 6; // a long wait plays in this long
const MIN_WAIT_SECONDS = 15; // shorter waits play as recorded, with no label

function newestVideo(dir) {
  const videos = readdirSync(dir, { recursive: true })
    .filter((file) => String(file).endsWith(".webm"))
    .map((file) => path.join(dir, String(file)));
  if (videos.length === 0) throw new Error(`No recording under ${dir}; run the demo first.`);
  return videos.sort((a, b) => statSync(b).mtimeMs - statSync(a).mtimeMs)[0];
}

function minutes(seconds) {
  return seconds < 90 ? `${Math.round(seconds)} s` : `${Math.round(seconds / 60)} min`;
}

/** The recording as pieces: [start, end or null for the rest, speed, label or null]. */
function pieces(waits) {
  const result = [];
  let at = 0;
  for (const wait of waits.filter((w) => w.end - w.start >= MIN_WAIT_SECONDS)) {
    result.push([at, wait.start, 1, null]);
    const taken = wait.end - wait.start;
    const speed = taken / SPED_UP_SECONDS;
    result.push([wait.start, wait.end, speed, `${wait.label}: ${minutes(taken)}, shown ${Math.round(speed)}x faster`]);
    at = wait.end;
  }
  result.push([at, null, 1, null]);
  return result;
}

function filterFor(parts) {
  mkdirSync(LABELS, { recursive: true });
  const chains = parts.map(([start, end, speed, label], index) => {
    const trim = end === null ? `trim=start=${start}` : `trim=start=${start}:end=${end}`;
    let chain = `[0:v]${trim},setpts=(PTS-STARTPTS)/${speed}`;
    if (label) {
      const file = path.join(LABELS, `${index}.txt`);
      writeFileSync(file, label);
      chain += `,drawtext=textfile='${file}':font='Noto Sans':fontsize=26:fontcolor=white`
        + ":box=1:boxcolor=0x14213Dd9:boxborderw=14:x=(w-text_w)/2:y=h-text_h-40";
    }
    return `${chain}[v${index}]`;
  });
  const inputs = parts.map((_, index) => `[v${index}]`).join("");
  return `${chains.join(";")};${inputs}concat=n=${parts.length}:v=1:a=0[out]`;
}

const video = newestVideo(path.join(OUTPUT, "playwright"));
const waits = JSON.parse(readFileSync(path.join(OUTPUT, "waits.json"), "utf8"));
const target = path.join(OUTPUT, "career-advisor-demo.mp4");
execFileSync("ffmpeg", [
  "-y", "-loglevel", "error", "-i", video,
  "-filter_complex", filterFor(pieces(waits)), "-map", "[out]",
  "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", target,
], { stdio: "inherit" });
console.log(`Wrote ${target}`);
