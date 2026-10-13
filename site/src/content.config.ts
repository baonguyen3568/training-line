import { docsLoader } from "@astrojs/starlight/loaders";
import { docsSchema } from "@astrojs/starlight/schema";
import { glob } from "astro/loaders";
import { z } from "astro/zod";
import { defineCollection } from "astro:content";
import { legalSlug } from "./data/legal";

// Every page lists the repo files its claims and numbers come from (ADR, benchmarks, ...).
// scripts/qa/check_site_numbers.py (bao-tester, run by check.sh) checks every path exists and every number on
// the page appears in its sources: the "no made-up numbers" gate. Numbers that need no source (e.g. "3 bước")
// are listed in numbers_without_source.
const sources = z.array(z.string()).default([]);
const numbers_without_source = z.array(z.string()).default([]);

export const collections = {
  docs: defineCollection({ loader: docsLoader(), schema: docsSchema({ extend: z.object({ sources, numbers_without_source }) }) }),
  // Marketing-layout MDX pages: policies, changelog, FAQ, ...
  pages: defineCollection({
    loader: glob({ pattern: "**/*.mdx", base: "./src/content/pages" }),
    schema: z.object({ title: z.string(), description: z.string(), sources, numbers_without_source }),
  }),
  // Policy pages: imported from docs/legal (owner: bao-security), rendered at /chinh-sach/<slug>/.
  legal: defineCollection({
    loader: glob({
      pattern: ["*.md", "!README.md"],
      base: "../docs/legal",
      generateId: ({ entry }) => legalSlug(entry),
    }),
    schema: z.object({ title: z.string(), description: z.string(), effective_date: z.string().optional() }),
  }),
};
