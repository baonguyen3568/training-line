// Content tables scroll horizontally (Starlight: table { display: block; overflow: auto }), so keyboard users must
// be able to focus them to scroll (axe scrollable-region-focusable, WCAG 2.1.1). Sätteri hast plugin (Astro 7's
// default Markdown processor; the deprecated markdown.rehypePlugins is not applied by it).
export const focusableTables = {
  name: "focusable-tables",
  element: [
    {
      filter: ["table"],
      visit(node, ctx) {
        ctx.setProperty(node, "tabindex", "0");
      },
    },
  ],
};
