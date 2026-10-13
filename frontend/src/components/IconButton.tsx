import type { ButtonHTMLAttributes } from "react";
import { Icon } from "./Icon";

/**
 * Icon-only control (DESIGN.md "Icon buttons"): the accessible name IS the tooltip text, plus the shortcut if
 * any ("Lưu · Ctrl+S"). Tooltip: CSS, ~400ms on hover, immediate on keyboard focus. Hit area ≥ 32×32.
 */
export function IconButton({
  icon,
  label,
  shortcut,
  tipSide = "down",
  size = 16,
  ...rest
}: {
  icon: string;
  label: string;
  shortcut?: string;
  tipSide?: "down" | "right" | "up";
  size?: number;
} & Omit<ButtonHTMLAttributes<HTMLButtonElement>, "aria-label" | "children">) {
  const tip = shortcut ? `${label} · ${shortcut}` : label;
  return (
    <button
      type="button"
      className="ib"
      aria-label={label}
      data-tip={tip}
      data-tip-side={tipSide}
      {...rest}
    >
      <Icon name={icon} size={size} />
    </button>
  );
}
