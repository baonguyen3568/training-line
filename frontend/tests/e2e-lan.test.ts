import { describe, expect, it } from "vitest";
import { isPrivateIPv4, isSpkiHash } from "../e2e/lan";

describe("external E2E tier: LAN target", () => {
  it("accepts only RFC 1918 addresses", () => {
    for (const ip of ["192.168.2.178", "10.0.0.1", "172.16.0.1", "172.31.255.255"]) {
      expect(isPrivateIPv4(ip), ip).toBe(true);
    }
    for (const ip of [
      "172.217.1.1",
      "192.0.2.1",
      "172.15.0.1",
      "172.32.0.1",
      "8.8.8.8",
      "192.168.1.256",
      "10.0.0",
      "127.0.0.1",
      " 10.0.0.1",
    ]) {
      expect(isPrivateIPv4(ip), ip).toBe(false);
    }
  });

  it("takes a base64 sha256 SPKI hash only", () => {
    expect(isSpkiHash(`${"A".repeat(43)}=`)).toBe(true);
    expect(isSpkiHash("abc")).toBe(false);
    expect(isSpkiHash(`${"A".repeat(43)}=,x`)).toBe(false);
  });
});
