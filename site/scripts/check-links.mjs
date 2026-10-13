// Fail the build when a built page links to a same-site path that was not generated.
// Dependency-free: scans dist/**/*.html for href/src="/..." values.
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join, resolve } from "node:path";

const dist = resolve(import.meta.dirname, "../dist");
const pages = [];
const walk = (dir) => {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) walk(p);
    else if (p.endsWith(".html")) pages.push(p);
  }
};
walk(dist);

const target = (url) => {
  const path = decodeURI(url.split(/[?#]/)[0]);
  const file = join(dist, path);
  return path.endsWith("/") ? join(file, "index.html") : file;
};
const broken = [];
for (const page of pages) {
  const html = readFileSync(page, "utf8");
  for (const [, url] of html.matchAll(/(?:href|src)="(\/(?!\/)[^"]*)"/g)) {
    if (!existsSync(target(url))) broken.push(`${page.slice(dist.length)} → ${url}`);
  }
}
if (broken.length) {
  console.error(`check-links: ${broken.length} broken internal link(s):\n${broken.join("\n")}`);
  process.exit(1);
}
console.log(`check-links: ${pages.length} pages, all internal links resolve`);
