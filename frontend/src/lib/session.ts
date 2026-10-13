/** A reload this recent that still finds the session ended means no sign-in page is served (Vite dev): stop. */
const RELOAD_KEY = "vlm-session-reload-at";
const RELOAD_WINDOW_MS = 30_000;

/** Reload into the server's sign-in page, remembering when, so a page that comes back signed out does not loop. */
export function reloadForSignIn(): void {
  try {
    sessionStorage.setItem(RELOAD_KEY, String(Date.now()));
  } catch {
    /* no storage: the next end shows the card at once */
  }
  window.location.reload();
}

/**
 * D-8: the session ended → reload into the server's sign-in page once (the draft is flushed on pagehide); if a
 * reload already happened just now, `show` the "open the link from the app's log" card instead of reloading again.
 * A 4401 that lands before the sign-out's own 200 reloads the same page twice at most: still one navigation.
 */
export function handleSessionEnded(show: () => void): void {
  let last: number;
  try {
    last = Number(sessionStorage.getItem(RELOAD_KEY) ?? 0);
  } catch {
    last = Date.now(); // no storage: never risk a reload loop
  }
  if (Date.now() - last < RELOAD_WINDOW_MS) show();
  else reloadForSignIn();
}
