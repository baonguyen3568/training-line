import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import type { AdvancedOption, ValidationIssue } from "../api/types";
import { t } from "../i18n";
import { isModelType, nearestUpstream } from "../lib/graph";
import { fold } from "../lib/search";
import { useWorkflow } from "../store/workflow";
import { FieldError } from "./FieldError";
import { Icon } from "./Icon";

type Values = Record<string, unknown>;

/** The value as people type it back: lists comma-separated, everything else as written. */
const show = (v: unknown) =>
  Array.isArray(v) ? v.join(", ") : v === null || v === undefined ? "" : String(v);

/** One option's control; `onChange(undefined)` = back to the engine's default (the key leaves the dict). */
function Control({
  opt,
  value,
  onChange,
}: {
  opt: AdvancedOption;
  value: unknown;
  onChange: (v: unknown) => void;
}) {
  const id = `adv-${opt.key}`;
  if (opt.type === "bool")
    return (
      <input
        id={id}
        type="checkbox"
        checked={Boolean(value ?? opt.default)}
        onChange={(e) => onChange(e.target.checked)}
      />
    );
  if (opt.type === "enum")
    return (
      <select
        id={id}
        value={value === undefined ? "" : String(value)}
        onChange={(e) =>
          onChange(
            e.target.value === ""
              ? undefined
              : (opt.choices ?? []).find((c) => String(c) === e.target.value),
          )
        }
      >
        <option value="">
          {opt.default === null ? t("advanced_panel.engine_default") : show(opt.default)}
        </option>
        {(opt.choices ?? []).map((c) => (
          <option key={String(c)} value={String(c)}>
            {String(c)}
          </option>
        ))}
      </select>
    );
  if (opt.type === "int" || opt.type === "float")
    return (
      <input
        id={id}
        type="number"
        inputMode={opt.type === "int" ? "numeric" : "decimal"}
        step={opt.type === "int" ? 1 : "any"}
        min={opt.min ?? undefined}
        max={opt.max ?? undefined}
        placeholder={opt.default === null ? t("advanced_panel.engine_default") : show(opt.default)}
        value={value === undefined ? "" : String(value)}
        onChange={(e) => {
          const raw = e.target.value;
          if (raw === "") return onChange(undefined);
          const n = Number(raw);
          onChange(Number.isFinite(n) ? (opt.type === "int" ? Math.trunc(n) : n) : raw);
        }}
      />
    );
  // list: comma-separated numbers (the server checks type and length)
  return (
    <input
      id={id}
      type="text"
      spellCheck={false}
      placeholder={t("advanced_panel.list_hint")}
      value={show(value)}
      onChange={(e) => {
        const parts = e.target.value
          .split(",")
          .map((p) => p.trim())
          .filter(Boolean);
        onChange(
          parts.length ? parts.map((p) => (Number.isFinite(Number(p)) ? Number(p) : p)) : undefined,
        );
      }}
    />
  );
}

/**
 * "Tuỳ chọn engine" (P016 §3, design 5ae3674): the train node's `advanced` param, {dotted key: value} holding only
 * what the user changed. Options come from the server, filtered to the wired model's template; keys stay as the
 * engine names them (mono, untranslated). A changed row is bold with a reset and an accessible "changed" note, a
 * group with a change or an error starts open, and a key's CFG-005 shows under its row.
 */
export function AdvancedOptions({
  nodeId,
  typeId,
  issues,
}: {
  nodeId: string;
  typeId: string;
  issues: ValidationIssue[];
}) {
  const workflow = useWorkflow((s) => s.workflow);
  const node = workflow?.nodes.find((n) => n.id === nodeId);
  const values = (node?.params.advanced ?? {}) as Values;
  const modelNode = nearestUpstream(workflow, nodeId, isModelType);
  const model = modelNode
    ? ((workflow?.nodes.find((n) => n.id === modelNode)?.params.model as string | undefined) ??
      null)
    : null;
  const [options, setOptions] = useState<AdvancedOption[] | null>(null);
  const [failed, setFailed] = useState(false);
  const [query, setQuery] = useState("");

  useEffect(() => {
    let live = true;
    api.advancedOptions(typeId, model).then(
      (r) => live && setOptions(r.options),
      () => live && setFailed(true),
    );
    return () => {
      live = false;
    };
  }, [typeId, model]);

  // CFG-005 of one key (params.key from the server); others show at the top
  const errorOf = (key: string) =>
    issues.find((i) => i.param === "advanced" && i.params.key === key)?.what;
  const generalErrors = issues.filter((i) => i.param === "advanced" && !i.params.key);

  const set = (key: string, v: unknown) => {
    const next = { ...values };
    if (v === undefined) delete next[key];
    else next[key] = v;
    useWorkflow.getState().setParam(nodeId, "advanced", next);
  };

  const shown = useMemo(() => {
    const q = fold(query.trim());
    return (options ?? []).filter((o) => !q || fold(o.key).includes(q));
  }, [options, query]);
  const groups = useMemo(() => {
    const m = new Map<string, AdvancedOption[]>();
    for (const o of shown) m.set(o.group, [...(m.get(o.group) ?? []), o]);
    return [...m.entries()];
  }, [shown]);

  if (failed) return <p role="alert">{t("app.catalog_load_failed")}</p>;
  if (!options) return <div className="skeleton-grid" aria-busy="true" />;

  return (
    <div className="adv-options">
      {!model && <p className="muted small">{t("advanced_panel.no_model")}</p>}
      {generalErrors.map((e) => (
        <FieldError key={e.what}>{e.what}</FieldError>
      ))}
      {options.length === 0 ? (
        <p className="muted">{t("advanced_panel.none")}</p>
      ) : (
        <>
          <label className="adv-search">
            <Icon name="search" size={14} />
            <input
              type="search"
              aria-label={t("advanced_panel.search")}
              placeholder={t("advanced_panel.search")}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </label>
          {groups.length === 0 && (
            <p className="muted small">{t("advanced_panel.no_match", { query })}</p>
          )}
          {groups.map(([group, opts]) => {
            const changed = opts.filter((o) => o.key in values).length;
            const hasError = opts.some((o) => errorOf(o.key));
            return (
              <details key={group} className="adv-group" open={changed > 0 || hasError || !!query}>
                <summary>
                  <span className="chev" aria-hidden="true">
                    <Icon name="chevron-right" size={14} />
                  </span>
                  <code>{group}</code>
                  {changed > 0 && (
                    <span className="muted small">
                      {" "}
                      · {t("advanced_panel.changed", { changed_count: changed })}
                    </span>
                  )}
                </summary>
                <ul>
                  {opts.map((o) => {
                    const isChanged = o.key in values;
                    const err = errorOf(o.key);
                    const popId = `adv-help-${o.key.replace(/\W+/g, "-")}`;
                    return (
                      <li key={o.key} className={isChanged ? "adv-row changed" : "adv-row"}>
                        <label
                          htmlFor={`adv-${o.key}`}
                          aria-description={
                            isChanged
                              ? t("advanced_panel.changed", { changed_count: 1 })
                              : undefined
                          }
                        >
                          <code>{o.key.slice(group.length + 1) || o.key}</code>
                        </label>
                        <Control opt={o} value={values[o.key]} onChange={(v) => set(o.key, v)} />
                        <button
                          type="button"
                          className="help-btn inline"
                          popoverTarget={popId}
                          aria-label={o.key}
                        >
                          <Icon name="circle-question-mark" size={14} />
                        </button>
                        <span id={popId} popover="auto" className="pop">
                          <code>{o.key}</code>
                          <span className="pop-p">
                            {o.type}
                            {o.min != null && o.max != null
                              ? ` · ${t("advanced_panel.range", { min: o.min, max: o.max })}`
                              : ""}
                          </span>
                          <span className="pop-p">
                            {t("advanced_panel.default", {
                              value:
                                o.default === null
                                  ? t("advanced_panel.engine_default")
                                  : show(o.default),
                            })}
                          </span>
                        </span>
                        {isChanged ? (
                          <button
                            type="button"
                            className="adv-reset"
                            aria-label={t("action.reset_default")}
                            title={t("action.reset_default")}
                            onClick={() => set(o.key, undefined)}
                          >
                            <Icon name="rotate-ccw" size={14} />
                          </button>
                        ) : (
                          <span className="adv-reset-slot" aria-hidden="true" />
                        )}
                        {err && (
                          <p className="adv-error" role="alert">
                            <span className="si si-bad" aria-hidden="true">
                              <Icon name="triangle-alert" size={14} />
                            </span>{" "}
                            {err}
                          </p>
                        )}
                      </li>
                    );
                  })}
                </ul>
              </details>
            );
          })}
        </>
      )}
    </div>
  );
}
