// G6 (P011): with prefers-reduced-motion: reduce, nothing keeps moving and nothing stays hidden waiting for an
// animation that will not run (a reveal-on-scroll left at opacity 0 is invisible content).
import { expect, sitePages, test } from "./fixtures";

for (const path of sitePages()) {
  test(`reduced motion ${path}`, async ({ page }) => {
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.goto(path);
    await page.mouse.wheel(0, 4000); // trigger scroll-driven reveals, if any
    await page.waitForTimeout(300);
    const moving = await page.evaluate(() =>
      document
        .getAnimations()
        .filter((a) => a.playState === "running")
        .map((a) => {
          const t = a.effect?.getComputedTiming();
          const target = (a.effect as KeyframeEffect | null)?.target as Element | null;
          return { duration: Number(t?.duration ?? 0), iterations: t?.iterations, on: target?.tagName ?? "?" };
        })
        .filter((a) => a.duration > 10 || a.iterations === Infinity),
    );
    expect(moving, "animations still running under reduced motion").toEqual([]);
    const autoplay = await page.locator("video[autoplay]:not([paused])").count();
    expect(autoplay, "autoplaying video under reduced motion").toBe(0);
    const hidden = await page.evaluate(() =>
      [...document.querySelectorAll('[data-testid^="motion-"]')]
        .filter((el) => Number(getComputedStyle(el).opacity) < 0.01 || getComputedStyle(el).visibility === "hidden")
        .map((el) => el.getAttribute("data-testid")),
    );
    expect(hidden, "motion-* content left invisible").toEqual([]);
  });
}
