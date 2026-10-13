// Static marketing + docs site (M2-3). Static output only, no CDN, no telemetry (set in npm scripts),
// CSP via Astro's hashed <meta> policy. Docs are served by Starlight under /docs/.
import { satteri } from "@astrojs/markdown-satteri";
import starlight from "@astrojs/starlight";
import { defineConfig, passthroughImageService } from "astro/config";
import { productName } from "./src/data/product.mjs";
import { focusableTables } from "./src/lib/focusable-tables.mjs";

export default defineConfig({
  site: "http://localhost",
  output: "static",
  trailingSlash: "always",
  image: { service: passthroughImageService() }, // no sharp until we actually need it
  markdown: { processor: satteri({ hastPlugins: [focusableTables] }) },
  security: {
    csp: {
      directives: [
        "default-src 'self'",
        "img-src 'self' data:",
        "font-src 'self'",
        "connect-src 'self'",
        "object-src 'none'",
        "base-uri 'self'",
        "form-action 'none'", // the site has no forms
      ],
      // Starlight's `is:inline` scripts (theme, mobile menu, pickers, sidebar state) are not hashed by Astro.
      // They change when Starlight is bumped; `site_check` (bao-security, run in `npm run check`) fails on drift.
      // Starlight and Expressive Code set CSS custom properties through style="" attributes (icon sizes, sidebar
      // depth, code colors). style-src-attr 'unsafe-inline' allows ONLY those attributes; <style> elements stay
      // hashed and scripts stay strict (no 'unsafe-inline' for scripts).
      styleDirective: { resources: ["'self'", { kind: "attribute", resource: "'unsafe-inline'" }] },
      scriptDirective: {
        hashes: [
          "sha256-VWo5Wp4aqSj6nSgMpeAp9cKieaoIfwFUAunAVugI5gA=",
          "sha256-f/zAUE74ucc3JYp4r4QQvkJofoQdkOIhHYK+jeZ6eko=",
          "sha256-GkZBRnvSuhtx/cvzvukVkX2JJZW+DdPlVr7BX8Tefqo=",
          "sha256-wX2yOADeV+NMngflD5uYi3vl50SHC4sfM1EmylVjlX4=",
          "sha256-7eCV4jtsr4t4knb3c4FCRPeu7GGZeOUGE3XvWix0XOQ=",
        ],
      },
    },
  },
  integrations: [
    starlight({
      title: productName,
      defaultLocale: "root",
      locales: { root: { label: "Tiếng Việt", lang: "vi" } },
      customCss: ["./src/generated/tokens.css", "./src/styles/fonts.css", "./src/styles/starlight.css"],
      disable404Route: true, // the marketing 404 page covers the whole site
      lastUpdated: false,
      sidebar: [
        {
          label: "Tài liệu",
          items: [{ autogenerate: { directory: "docs" } }, { label: "Tra mã lỗi", link: "/docs/ma-loi/" }],
        },
      ],
    }),
  ],
});
