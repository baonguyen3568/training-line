import type { ReactNode } from "react";
import { Icon } from "./Icon";

/**
 * A field's error under it (design D7: no side tab): circle-x in critical status ink, the words in text tokens.
 * `id` lets the field point at it with aria-describedby (read when the field is focused). `live` (role="alert") only
 * for an error that follows an action (a save, a server refusal): one said while typing would be read again on
 * every key.
 */
export function FieldError({
  id,
  live = false,
  children,
}: {
  id?: string;
  live?: boolean;
  children: ReactNode;
}) {
  return (
    <p id={id} className="field-error" role={live ? "alert" : undefined}>
      <span className="si si-bad" aria-hidden="true">
        <Icon name="circle-x" size={14} />
      </span>
      <span>{children}</span>
    </p>
  );
}
