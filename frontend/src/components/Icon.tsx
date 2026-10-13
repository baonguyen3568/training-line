import { createElement } from "react";
import { ICONS } from "../generated/icons";

/**
 * Lucide icon (ISC). Decorative by default; pass `label` only when the icon is the whole content of an element
 * that has no other accessible name. Unknown names render nothing (and fail the icon test).
 */
export function Icon({ name, size = 16, label }: { name: string; size?: number; label?: string }) {
  const nodes = ICONS[name];
  if (!nodes) return null;
  return (
    <svg
      className="ico"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.75}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden={label ? undefined : true}
      role={label ? "img" : undefined}
      aria-label={label}
    >
      {nodes.map(([tag, attrs], i) => createElement(tag, { key: i, ...attrs }))}
    </svg>
  );
}
