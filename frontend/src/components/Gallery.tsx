import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import type { TemplateMeta } from "../api/types";
import { t, tp } from "../i18n";
import { formatEstimate } from "../lib/format";

type Filter = "all" | "finetune" | "quantize" | "eval";
const FILTERS: Filter[] = ["all", "finetune", "quantize", "eval"];

export function Gallery({ onOpen }: { onOpen: (id: string) => void }) {
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
      <div className="gallery-tools">
        <input
          type="search"
          placeholder={t("gallery.search")}
          aria-label={t("gallery.search")}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <div className="segmented" role="radiogroup" aria-label="filter">
          {FILTERS.map((f) => (
            <button
              key={f}
              type="button"
              role="radio"
              aria-checked={filter === f}
              className={filter === f ? "on" : ""}
              onClick={() => setFilter(f)}
            >
              {t(`gallery.filter_${f}`)}
            </button>
          ))}
        </div>
      </div>
      {error && <p role="alert">{t("error.SYS-001.title")}</p>}
      {!items && !error && <div className="skeleton-grid" aria-busy="true" />}
      {items && shown.length === 0 && <p className="muted">{t("gallery.empty")}</p>}
      <ul className="card-grid">
        {shown.map((it) => (
          <li key={it.id} className="card">
            <h2>{tp(it.title_key.split("."))}</h2>
            <p>{tp(it.description_key.split("."))}</p>
            <p className="meta">
              ⏱ {formatEstimate(it.estimate.minutes, "minutes", it.estimate.source)} ·{" "}
              {formatEstimate(it.estimate.ram_gb, "ram_gb", it.estimate.source)}
            </p>
            <p className="meta">
              ● {t(`gallery.level_${it.difficulty}`)}
              {it.sample_data && <> · {t("gallery.sample_data_included")} ✓</>}
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
