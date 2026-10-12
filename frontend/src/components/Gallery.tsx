import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import type { TemplateMeta } from "../api/types";
import { t, tp, tpOptional } from "../i18n";
import { formatEstimate } from "../lib/format";

/** One "no measurement yet" instead of repeating it per field (design review r4). */
function estimateLine(it: TemplateMeta): string {
  const e = it.estimate;
  if (e.source === "unknown" || (e.minutes == null && e.ram_gb == null))
    return `⏱ ${t("estimate.source.unknown")}`;
  return `⏱ ${formatEstimate(e.minutes, "minutes", e.source)} · ${formatEstimate(e.ram_gb, "ram_gb", e.source)}`;
}

type Filter = "all" | "finetune" | "quantize" | "eval";
const FILTERS: Filter[] = ["all", "finetune", "quantize", "eval"];

export function Gallery({
  onOpen,
  onBlank,
}: {
  onOpen: (id: string) => void;
  onBlank: () => void;
}) {
  const [items, setItems] = useState<TemplateMeta[] | null>(null);
  const [error, setError] = useState(false);
  const [filter, setFilter] = useState<Filter>("all");
  const [query, setQuery] = useState("");

  useEffect(() => {
    api.templates().then(setItems, () => setError(true));
  }, []);

  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    return (items ?? []).filter((it) => {
      if (filter !== "all" && !it.tags.includes(filter)) return false;
      if (!q) return true;
      const text =
        `${tp(it.title_key.split("."))} ${tp(it.description_key.split("."))}`.toLowerCase();
      return text.includes(q);
    });
  }, [items, filter, query]);

  return (
    <main className="gallery">
      <h1>{t("gallery.title")}</h1>
      <p className="muted">{t("gallery.subtitle")}</p>
      <button type="button" onClick={onBlank}>
        ＋ {t("gallery.blank_workflow")}
      </button>
      <div className="gallery-tools">
        <input
          type="search"
          placeholder={t("gallery.search")}
          aria-label={t("gallery.search")}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <div className="segmented">
          {FILTERS.map((f) => (
            <button
              key={f}
              type="button"
              aria-pressed={filter === f}
              className={filter === f ? "on" : ""}
              onClick={() => setFilter(f)}
            >
              {t(`gallery.filter_${f}`)}
            </button>
          ))}
        </div>
      </div>
      {error && <p role="alert">{t("app.catalog_load_failed")}</p>}
      {!items && !error && <div className="skeleton-grid" aria-busy="true" />}
      {items && shown.length === 0 && <p className="muted">{t("gallery.empty")}</p>}
      <ul className="card-grid">
        {shown.map((it) => (
          <li key={it.id} className="card">
            <h2>{tp(it.title_key.split("."))}</h2>
            <p>{tp(it.description_key.split("."))}</p>
            {tpOptional(["template", it.id, "built_on"]) && (
              <p className="meta">{tpOptional(["template", it.id, "built_on"])}</p>
            )}
            <p className="meta">{estimateLine(it)}</p>
            <p className="meta">
              ● {t(`gallery.level_${it.difficulty}`)}
              {it.sample_data && <> · {t("gallery.sample_data_included")} ✓</>}
              {it.requires_hf_account && (
                <span className="chip needs-account">
                  {" "}
                  🔑 {tp(["template", it.id, "needs_account"])}
                </span>
              )}
            </p>
            <button type="button" className="primary" onClick={() => onOpen(it.id)}>
              {t("gallery.use_template")}
            </button>
          </li>
        ))}
      </ul>
      <p className="muted small">{t("gallery.opens_copy")}</p>
    </main>
  );
}
