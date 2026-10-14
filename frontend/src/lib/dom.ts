/** A value for a quoted CSS attribute selector (`[id="…"]`); CSS.escape is missing in some environments (jsdom). */
export const attrValue = (v: string) => v.replace(/["\\]/g, "\\$&");
