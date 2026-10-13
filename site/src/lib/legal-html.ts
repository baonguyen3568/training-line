import { productName } from "~/data/product.mjs";
import { legalSlug } from "~/data/legal";

// docs/legal is written before the product has a name: {{PRODUCT_NAME}} is filled in at build time.
export function withProductName(text: string): string {
  return text.replaceAll("{{PRODUCT_NAME}}", productName);
}

// docs/legal links to its siblings as `./<file>.md` (readable on GitHub); on the site they are /chinh-sach/<slug>/.
// An unmapped file throws in legalSlug, so a new cross-link can never ship as a dead .md link.
const SIBLING_LINK = /href="(?:\.\/)?([a-z0-9-]+\.md)(#[^"]*)?"/g;

export function withSiteLinks(html: string): string {
  return html.replace(SIBLING_LINK, (_, file: string, hash = "") => `href="/chinh-sach/${legalSlug(file)}/${hash}"`);
}
