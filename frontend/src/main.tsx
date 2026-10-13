import "@xyflow/react/dist/style.css";
import "./generated/tokens.css";
import "./styles/fonts.css";
import "./styles/d4.css";
import "./styles/app.css";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { installPopoverAnchoring } from "./lib/popoverAnchor";

installPopoverAnchoring();

const root = document.getElementById("root");
if (root) {
  createRoot(root).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
}
