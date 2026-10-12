import { useEffect, useState } from "react";
import { api } from "../api";
import type { SystemStatus } from "../api/types";
import { t } from "../i18n";

const POLL_MS = 10_000;

export type Nav = "home" | "runs";

export function Header({ onNavigate, active }: { onNavigate: (to: Nav) => void; active: Nav }) {
  const [sys, setSys] = useState<SystemStatus | null>(null);

  useEffect(() => {
    let alive = true;
    const load = () =>
      api
        .system()
        .then((s) => alive && setSys(s))
        .catch(() => alive && setSys(null));
    load();
    const timer = setInterval(load, POLL_MS);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);

  const lock = sys?.gpu_lock;
  const gpuText = !lock
    ? "GPU · …"
    : lock.stale
      ? t("header.gpu_stale")
      : lock.held
        ? t("header.gpu_busy", {
            who: lock.who ?? "?",
            job: lock.job ?? "?",
            since: lock.since ?? "?",
          })
        : t("header.gpu_free");
  const gpuState = !lock ? "unknown" : lock.stale ? "critical" : lock.held ? "busy" : "free";

  return (
    <header className="app-header">
      <button type="button" className="brand" onClick={() => onNavigate("home")}>
        ▣ {t("app.name")}
      </button>
      <nav className="app-nav">
        <button
          type="button"
          aria-current={active === "home" ? "page" : undefined}
          onClick={() => onNavigate("home")}
        >
          {t("app.nav_workflows")}
        </button>
        <button
          type="button"
          aria-current={active === "runs" ? "page" : undefined}
          onClick={() => onNavigate("runs")}
        >
          {t("app.nav_runs")}
        </button>
      </nav>
      <div className="header-status">
        <span className={`chip gpu-${gpuState}`} title={gpuText}>
          {gpuText}
        </span>
        {sys && (
          <span className="chip">
            {/* Unified memory on GB10: show MemTotal/MemAvailable, never "GPU memory %" (P006 §7). */}
            {t("header.ram", {
              used_gb: Math.round(sys.mem_total_gb - sys.mem_available_gb),
              total_gb: Math.round(sys.mem_total_gb),
            })}{" "}
            · {t("header.ram_available", { avail_gb: Math.round(sys.mem_available_gb) })}
          </span>
        )}
        {sys && (
          <span className="chip">
            💾 {t("header.disk_free", { free_gb: Math.round(sys.disk_free_gb) })}
          </span>
        )}
      </div>
    </header>
  );
}
