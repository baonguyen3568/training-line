import {
  type CSSProperties,
  type ReactNode,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import data from "../generated/reference-pipelines.json";
import { t, tpOptional } from "../i18n";
import { formatMinutesRange } from "../lib/format";
import { licenseName } from "../lib/licenses";
import { portShape, tokenVar } from "../lib/ports";
import { formatScore } from "../lib/score";
import { fold } from "../lib/search";
import { useWorkflow } from "../store/workflow";
import { ArchDiagram, ArchLegend, type TrainArch } from "./ArchDiagram";
import { Icon } from "./Icon";

type Text = { vi: string; en: string };
/** configs/reference_pipelines.yaml v2 (common/reference_pipelines.py): one pipeline per training METHOD (ADR-022). */
interface Pipeline {
  id: string;
  method: string;
  kind: "app_template" | "candidate" | "external";
  template_id: string | null;
  status: "available" | "planned" | "reference_only";
  difficulty: "basic" | "advanced" | null;
  title: Text;
  /** i18n key of the method's one "when to use" sentence: method.<method>.when */
  when_to_use_key: string;
  summary_key?: string;
  train_arch: TrainArch;
  source: { url: string; publisher: string; paper_url?: string | null };
  license: { code: string | null; model: string | null; data: string | null };
  model_parts?: { part: string; license: string }[];
  requires_hf_terms?: boolean;
  notice?: string | null;
  steps: { node: string | null; label: Text }[];
  /** The configuration the numbers were measured with; the method itself works with any model/data (ADR-021). */
  measured_with?: { model: string; data: string } | null;
  /** Numbers from a real run on this machine (research), with their source in the data file. */
  measured?: {
    train_minutes?: number | null;
    peak_ram_gb?: number | null;
    config?: string;
    /** compare_with: another pipeline scored on the same benchmark/metric (checked by the schema). */
    score?: {
      metric: string;
      value: number;
      ci95?: [number, number] | null;
      n?: number;
      compare_with?: string | null;
    } | null;
  } | null;
}
const pipelines = (data as unknown as { pipelines: Pipeline[] }).pipelines;
type Filter = "all" | Pipeline["kind"];
const FILTERS: Filter[] = ["all", "app_template", "external", "candidate"];

const lic = (id: string | null) => (id ? licenseName(id) : t("pipelines.see_source"));
/** One short licence line for the badge: code and model licences, each once; the details sit behind "?". */
const licenseLine = (p: Pipeline) =>
  [...new Set([p.license.code, p.license.model, p.license.data].map(lic))].join(" · ");
const host = (url: string) => {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
};

/**
 * The card's "Đo trên DGX Spark" line: research's measured run (data file, with its source) only. The server's time
 * is measured on the user's machine, which may not be a DGX Spark, so it belongs to the Inspector / workflow summary
 * ("đã đo trên máy này"), never to this card (design 1.0). Peak RAM joins when measured.
 */
function timeText(p: Pipeline): string {
  const m = p.measured;
  const range = m?.train_minutes ? { lo: m.train_minutes, hi: m.train_minutes } : null;
  const ram = m?.peak_ram_gb ? t("pipelines.ram_peak", { gb: m.peak_ram_gb }) : null;
  // a part that does not apply (no training, train_minutes null) is left out, never "no measurement yet"
  if (!range && ram) return ram;
  const time = formatMinutesRange(range);
  return ram ? `${time} · ${ram}` : time;
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
    other_value: formatScore(other.measured?.score?.metric ?? "", value),
  }).split("\u0000");
  return (
    <p>
      {before}
      <button type="button" className="link" onClick={() => onShow(other.id)}>
        {t(`method.${other.method}.title`)}
      </button>
      {after}
    </p>
  );
}

/** What the number means, first in its popover when design wrote it (pipelines.metric_help.<id>, design 94ee3e9). */
function MetricHelp({ score }: { score: { metric: string; n?: number } }) {
  const help = tpOptional(["pipelines", "metric_help", score.metric], { n: score.n ?? "" });
  return help ? <p>{help}</p> : null;
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
      (!q || fold(`${t(`method.${p.method}.title`)} ${p.title.vi} ${p.title.en}`).includes(q)),
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
              <header className="rp-card-head">
                <h2>{t(`method.${p.method}.title`)}</h2>
                {p.difficulty && (
                  <span className="chip">{t(`pipelines.level.${p.difficulty}`)}</span>
                )}
              </header>
              <p className="rp-when clamp2">{tpOptional(p.when_to_use_key.split(".")) ?? ""}</p>
              <div className="rp-arch">
                <ArchDiagram arch={p.train_arch} />
                <Help id={`arch-${p.id}`} title={t("arch.legend_title")}>
                  <ArchLegend />
                </Help>
              </div>
              {p.notice && <p className="rp-notice">{p.notice}</p>}
              <div className="rp-meta">
                <a href={p.source.url} target="_blank" rel="noopener noreferrer">
                  {p.source.publisher} <Icon name="external-link" size={12} />
                </a>
                <span className="chip">{licenseLine(p)}</span>
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
                {p.measured_with && (
                  <span className="muted">
                    {t("pipelines.measured_with", { model: p.measured_with.model })}
                  </span>
                )}
              </div>
              {p.measured && (
                <div className="rp-measured">
                  {t("pipelines.measured")}
                  {timeText(p) && ` · ${timeText(p)}`}
                  {p.measured.score && (
                    <>
                      {" · "}
                      {tpOptional(["pipelines", "metric", p.measured.score.metric]) ??
                        p.measured.score.metric}{" "}
                      {formatScore(p.measured.score.metric, p.measured.score.value)}
                    </>
                  )}
                  <Help id={`measured-${p.id}`} title={t("pipelines.measured")}>
                    {p.measured.score && <MetricHelp score={p.measured.score} />}
                    <ScoreCompare p={p} onShow={showCard} />
                    {p.measured.score && <p>{t("pipelines.score_single_run")}</p>}
                    {p.measured.config && <p className="small">{p.measured.config}</p>}
                    <p className="small">
                      <a href={p.source.url} target="_blank" rel="noopener noreferrer">
                        {p.source.publisher}
                      </a>
                      {p.source.paper_url && (
                        <>
                          {" · "}
                          <a href={p.source.paper_url} target="_blank" rel="noopener noreferrer">
                            {t("pipelines.paper")}
                          </a>
                        </>
                      )}
                    </p>
                    <p>{t("pipelines.swap_hint")}</p>
                  </Help>
                </div>
              )}
              {p.requires_hf_terms && (
                <div className="rp-gated" data-testid="needs-hf-account">
                  <Icon name="key-round" size={14} />
                  {t("pipelines.hf_terms")}
                  <Help id={`hf-${p.id}`} title={t("pipelines.hf_terms")}>
                    <p>{t("pipelines.hf_terms_detail")}</p>
                  </Help>
                </div>
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
                  href={p.source.url}
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
