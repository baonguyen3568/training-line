import type { ReactNode } from "react";
import { Icon } from "./Icon";

/**
 * A field's error under it (design D7: no side tab): circle-x in critical status ink, the words in text tokens.
 * `id` lets the field point at it with aria-describedby.
 */
export function FieldError({ id, children }: { id?: string; children: ReactNode }) {
  return (
    <p id={id} className="field-error" role="alert">
      <span className="si si-bad" aria-hidden="true">
        <Icon name="circle-x" size={14} />
      </span>
      <span>{children}</span>
    </p>
  );
}
