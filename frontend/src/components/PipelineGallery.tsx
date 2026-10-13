import {
  type CSSProperties,
  type ReactNode,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { api } from "../api";
import type { Range } from "../api/types";
import data from "../generated/reference-pipelines.json";
import { getLocale, t, tpOptional } from "../i18n";
import { formatMinutesRange } from "../lib/format";
import { portShape, tokenVar } from "../lib/ports";
import { fold } from "../lib/search";
import { useWorkflow } from "../store/workflow";
import { Icon } from "./Icon";

type Text = { vi: string; en: string };
interface Pipeline {
  id: string;
  kind: "app_template" | "candidate" | "external";
  template_id: string | null;
  status: "available" | "planned" | "reference_only";
  difficulty: "basic" | "advanced" | null;
  title: Text;
  source_url: string;
  license: { code: string | null; model: string | null; data: string | null };
  model_parts?: { part: string; license: string }[];
  requires_hf_terms?: boolean;
  notice?: string;
  steps: { node: string | null; label: Text }[];
  /** i18n key of the one-line "what you get" (design gallery r2 #3). */
  summary_key?: string;
  /** Numbers from a real run on this machine (research), with their source in the data file. */
  measured?: {
    train_minutes?: number;
    peak_ram_gb?: number;
    /** compare_with: another pipeline scored on the same benchmark/metric (checked by the generator). */
    score?: { metric: string; value: number; compare_with?: string };
  } | null;
}
const pipelines = (data as { pipelines: Pipeline[] }).pipelines;
type Filter = "all" | Pipeline["kind"];
const FILTERS: Filter[] = ["all", "app_template", "external", "candidate"];

// SPDX-ish ids from the data file → display names (unknown ids are shown as written, never invented).
const LICENSE_NAME: Record<string, string> = {
  "apache-2.0": "Apache-2.0",
  mit: "MIT",
  "cc-by-4.0": "CC-BY-4.0",
  "nvidia-open-model-license": "NVIDIA Open Model License",
  "openmdw-1.1": "OpenMDW-1.1",
};
const lic = (id: string | null) => (id ? (LICENSE_NAME[id] ?? id) : t("pipelines.see_source"));
const loc = (x: Text) => (getLocale() === "en" ? x.en : x.vi);
const host = (url: string) => {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
};

/**
 * Time on the card: a run research measured on this machine (data file, with its source) first, else the server's
 * template estimate, else "no measurement yet". Peak RAM joins when measured.
 */
function timeText(p: Pipeline, estimate: Range | null | undefined): string {
  const m = p.measured;
  const time = formatMinutesRange(
    m?.train_minutes ? { lo: m.train_minutes, hi: m.train_minutes } : (estimate ?? null),
  );
  return m?.peak_ram_gb ? `${time} · ${t("pipelines.ram_peak", { gb: m.peak_ram_gb })}` : time;
}

/**
 * 'Trên cùng bộ kiểm tra, pipeline "X" đạt N.' for a pipeline research compared on the same benchmark; X is a
 * button that scrolls to that card (design: the reader must find it).
 */
function ScoreCompare({ p, onShow }: { p: Pipeline; onShow: (id: string) => void }) {
  const other = pipelines.find((x) => x.id === p.measured?.score?.compare_with);
  const value = other?.measured?.score?.value;
  if (!other || value === undefined) return null;
  const [before, after = ""] = t("pipelines.score_compare", {
    other: "\u0000",
    other_value: value.toFixed(2),
  }).split("\u0000");
  return (
    <p>
      {before}
      <button type="button" className="link" onClick={() => onShow(other.id)}>
        {loc(other.title)}
      </button>
      {after}
    </p>
  );
}

/** "?" + native popover (DESIGN.md "?" Popover): secondary detail never sits on the card itself. */
function Help({ id, title, children }: { id: string; title: string; children: ReactNode }) {
  return (
    <>
      <button type="button" className="help-btn inline" popoverTarget={id} aria-label={title}>
        <Icon name="circle-question-mark" size={14} />
      </button>
      <div id={id} popover="auto" className="pop">
        <strong>{title}</strong>
        {children}
      </div>
    </>
  );
}

// Two rows of chips at most so every card has the same height (design gallery r1 #1); the rest folds into "+N".
const MAX_CHIPS = 5;

/**
 * At most two rows of chips (design gallery r2 #1): every card's title then starts at the same height. Chips that
 * would wrap to a third row fold into "+N bước", measured after layout (and again when the card resizes).
 */
function MiniDiagram({ steps }: { steps: Pipeline["steps"] }) {
  const specs = useWorkflow((s) => s.specs);
  const ports = useWorkflow((s) => s.ports);
  const ref = useRef<HTMLOListElement>(null);
  const [fit, setFit] = useState(Math.min(steps.length, MAX_CHIPS));
  const shown = fit < steps.length ? steps.slice(0, Math.max(1, fit - 1)) : steps;
  const rest = steps.slice(shown.length);
  useLayoutEffect(() => {
    const ol = ref.current;
    if (!ol) return;
    const rows = new Set([...ol.children].map((li) => (li as HTMLElement).offsetTop)).size;
    if (rows > 2 && fit > 1) setFit(fit - 1);
  });
  useEffect(() => {
    const ol = ref.current;
    if (!ol || typeof ResizeObserver === "undefined") return;
    let width = ol.clientWidth;
    const ro = new ResizeObserver(() => {
      if (ol.clientWidth === width) return;
      width = ol.clientWidth;
      setFit(Math.min(steps.length, MAX_CHIPS)); // re-measure from the start at the new width
    });
    ro.observe(ol);
    return () => ro.disconnect();
  }, [steps.length]);
  return (
    <ol className="mini" aria-hidden="true" ref={ref}>
      {shown.map((st, i) => {
        const spec = st.node ? specs.get(st.node) : undefined;
        const out = spec && Object.values(spec.outputs)[0];
        const pt = out && ports.find((p) => p.id === out.type);
        return (
          // biome-ignore lint/suspicious/noArrayIndexKey: steps are a fixed ordered list
          <li key={i} className={st.node ? "" : "no-node"}>
            {out && (
              <span
                className={`pg pg-${portShape(out.type)}`}
                style={
                  {
                    "--pc": pt ? tokenVar(pt.color_token) : "var(--color-port-value)",
                  } as CSSProperties
                }
              />
            )}
            {loc(st.label)}
          </li>
        );
      })}
      {rest.length > 0 && (
        <li className="more" title={rest.map((st) => loc(st.label)).join(" → ")}>
          {t("pipelines.more_steps", { step_count: rest.length })}
        </li>
      )}
    </ol>
  );
}

/**
 * D4 "Pipeline tham khảo" (P4-1, configs/reference_pipelines.yaml by bao-research): app templates open in the
 * editor; external and candidate pipelines link to their source. No measured number exists yet, so time always
 * reads "chưa có số đo" and an unassessed difficulty reads "chưa đánh giá".
 */
export function PipelineGallery() {
  const { openTemplate, newWorkflow } = useWorkflow.getState();
  const [filter, setFilter] = useState<Filter>("all");
  const [query, setQuery] = useState("");
  /** Show one card even if the tab or search hides it, then bring it into view (score comparison link). */
  const showCard = (id: string) => {
    setFilter("all");
    setQuery("");
    requestAnimationFrame(() => {
      const card = document.getElementById(`rp-${id}`);
      card?.scrollIntoView({ block: "center" });
      card?.focus({ preventScroll: true });
    });
  };
  // App templates carry the server's measured time (estimate.minutes); never a number from the data file.
  const [minutes, setMinutes] = useState<Map<string, Range | null>>(new Map());
  useEffect(() => {
    api.templates().then(
      (list) => setMinutes(new Map(list.map((x) => [x.id, x.estimate.minutes]))),
      () => {},
    );
  }, []);
  const counts = useMemo(() => {
    const c: Record<Filter, number> = {
      all: pipelines.length,
      app_template: 0,
      external: 0,
      candidate: 0,
    };
    for (const p of pipelines) c[p.kind]++;
    return c;
  }, []);
  const q = fold(query);
  const shown = pipelines.filter(
    (p) =>
      (filter === "all" || p.kind === filter) &&
      (!q || fold(`${p.title.vi} ${p.title.en}`).includes(q)),
  );

  return (
    <main className="rp-gallery">
      <header className="rp-head">
        <h1>{t("pipelines.title")}</h1>
        <div className="seg" role="tablist" aria-label={t("pipelines.title")}>
          {FILTERS.map((f) => (
            <button
              key={f}
              type="button"
              role="tab"
              aria-selected={filter === f}
              className={filter === f ? "on" : ""}
              onClick={() => setFilter(f)}
            >
              {t(`pipelines.filter.${f}`)} <span className="count">{counts[f]}</span>
            </button>
          ))}
        </div>
        <button type="button" className="btn ghost sm" onClick={newWorkflow}>
          <Icon name="file-plus" size={14} />
          {t("gallery.blank_workflow")}
        </button>
        <label className="rp-search">
          <Icon name="search" size={14} />
          <input
            type="search"
            placeholder={t("pipelines.search")}
            aria-label={t("pipelines.search")}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </label>
      </header>
      {shown.length === 0 && <p className="muted">{t("gallery.empty")}</p>}
      <ul className="rp-grid">
        {shown.map((p) => {
          const open = p.status === "available" && p.template_id;
          return (
            <li
              key={p.id}
              id={`rp-${p.id}`}
              tabIndex={-1}
              className={`rp-card ${p.kind}${p.status !== "available" ? " dim" : ""}`}
            >
              <MiniDiagram steps={p.steps} />
              <h2>{loc(p.title)}</h2>
              {p.summary_key && (
                <p className="rp-summary">{tpOptional(p.summary_key.split(".")) ?? ""}</p>
              )}
              {/* where it comes from; the status (candidate, planned…) is said once, in the footer */}
              <p className="rp-src">
                {p.kind === "app_template" ? (
                  <>
                    <Icon name="boxes" size={14} /> {t("pipelines.kind.app_template")}
                  </>
                ) : (
                  <>
                    <Icon name="link" size={14} /> {host(p.source_url)}
                  </>
                )}
              </p>
              {p.notice && <p className="rp-notice">{p.notice}</p>}
              <dl>
                <div>
                  <dt>
                    {t("pipelines.license")}
                    <Help id={`lic-${p.id}`} title={t("pipelines.license_detail")}>
                      <p>
                        {t("pipelines.license_parts", {
                          code: lic(p.license.code),
                          model: lic(p.license.model),
                          data: lic(p.license.data),
                        })}
                      </p>
                      {p.model_parts?.map((m) => (
                        <p key={m.part}>
                          {m.part}: {lic(m.license)}
                        </p>
                      ))}
                    </Help>
                  </dt>
                  {/* one short line on the card (design gallery r1 B1): the three licences, details behind "?" */}
                  <dd>
                    {[...new Set([p.license.code, p.license.model, p.license.data].map(lic))].join(
                      " · ",
                    )}
                  </dd>
                </div>
                <div>
                  <dt>{t("pipelines.difficulty")}</dt>
                  <dd>
                    {p.difficulty
                      ? t(`pipelines.level.${p.difficulty}`)
                      : t("pipelines.not_assessed")}
                  </dd>
                </div>
                <div>
                  <dt>{t("pipelines.time")}</dt>
                  <dd>
                    {timeText(p, p.template_id ? minutes.get(p.template_id) : undefined)}
                    {p.measured?.score && (
                      <>
                        {" · "}
                        {tpOptional(["pipelines", "metric", p.measured.score.metric]) ??
                          p.measured.score.metric}{" "}
                        {p.measured.score.value.toFixed(2)}
                        <Help
                          id={`score-${p.id}`}
                          title={
                            tpOptional(["pipelines", "metric", p.measured.score.metric]) ??
                            p.measured.score.metric
                          }
                        >
                          <ScoreCompare p={p} onShow={showCard} />
                          <p>{t("pipelines.score_single_run")}</p>
                        </Help>
                      </>
                    )}
                  </dd>
                </div>
              </dl>
              {p.requires_hf_terms && (
                <p className="rp-gated" data-testid="needs-hf-account">
                  <Icon name="key-round" size={14} />
                  {t("pipelines.hf_terms")}
                  <Help id={`hf-${p.id}`} title={t("pipelines.hf_terms")}>
                    <p>{t("pipelines.hf_terms_detail")}</p>
                  </Help>
                </p>
              )}
              <div className="rp-actions">
                {open ? (
                  <button
                    type="button"
                    className="btn primary sm"
                    onClick={() => void openTemplate(p.template_id as string)}
                  >
                    {t("pipelines.open")}
                  </button>
                ) : (
                  <span className="rp-status">
                    <Icon name={p.status === "planned" ? "clock" : "book-open"} size={14} />
                    {t(`pipelines.status.${p.status}`)}
                  </span>
                )}
                <a
                  className="btn ghost sm"
                  href={p.source_url}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Icon name="external-link" size={14} />
                  {t("pipelines.view_source")}
                </a>
              </div>
            </li>
          );
        })}
      </ul>
    </main>
  );
}
