// LAN edge target check for the external E2E tier (bao-security review): only an RFC 1918 address may be a target,
// since that run pins the edge's own certificate instead of a public CA.

/** True for 10.0.0.0/8, 172.16.0.0/12 and 192.168.0.0/16 written as four decimal octets 0-255. */
export function isPrivateIPv4(ip: string): boolean {
  const m = /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/.exec(ip);
  if (!m) return false;
  const [a, b, c, d] = m.slice(1).map(Number) as [number, number, number, number];
  if ([a, b, c, d].some((o) => o > 255)) return false;
  return a === 10 || (a === 172 && b >= 16 && b <= 31) || (a === 192 && b === 168);
}

/** A base64 SHA-256 SPKI hash (Chromium's --ignore-certificate-errors-spki-list format). */
export function isSpkiHash(value: string): boolean {
  return /^[A-Za-z0-9+/]{43}=$/.test(value);
}
