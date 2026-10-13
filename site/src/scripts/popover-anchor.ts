// Anchor every native "?" popover to the button that opened it (design gallery r2): native popovers live in the
// top layer and centre on the page, so a card's licence detail could sit over the NEXT card. Framework-free; the
// app has a twin in frontend/src/lib/popoverAnchor.ts. Under 8px from the trigger, flipped near viewport edges.

const GAP = 8;
const EDGE = 8;
let invoker: HTMLElement | null = null;

export function placePopover(pop: HTMLElement, trigger: HTMLElement): void {
  const r = trigger.getBoundingClientRect();
  pop.style.position = "fixed";
  pop.style.margin = "0";
  pop.style.inset = "auto";
  const w = pop.offsetWidth;
  const h = pop.offsetHeight;
  const below = r.bottom + GAP + h <= window.innerHeight - EDGE;
  const top = below ? r.bottom + GAP : Math.max(EDGE, r.top - GAP - h);
  const left = Math.min(Math.max(EDGE, r.left + r.width / 2 - 20), window.innerWidth - w - EDGE);
  pop.style.top = `${top}px`;
  pop.style.left = `${left}px`;
  pop.dataset.side = below ? "below" : "above";
  // arrow points at the trigger's centre
  pop.style.setProperty("--arrow-x", `${Math.min(Math.max(12, r.left + r.width / 2 - left), w - 12)}px`);
}

/** Install once: remember the clicked trigger, place its popover when it opens. */
export function installPopoverAnchoring(doc: Document = document): void {
  doc.addEventListener(
    "click",
    (e) => {
      const t = (e.target as Element | null)?.closest?.("[popovertarget]");
      if (t instanceof HTMLElement) invoker = t;
    },
    true,
  );
  doc.addEventListener(
    "toggle",
    (e) => {
      const pop = e.target as HTMLElement;
      if (!(pop instanceof HTMLElement) || !pop.matches(":popover-open")) return;
      const trigger =
        invoker?.getAttribute("popovertarget") === pop.id
          ? invoker
          : doc.querySelector<HTMLElement>(`[popovertarget="${CSS.escape(pop.id)}"]`);
      if (trigger) placePopover(pop, trigger);
    },
    true,
  );
}

installPopoverAnchoring();
