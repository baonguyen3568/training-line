// H-prompt (P021): the trained question is a field hint (field_hint MODEL-007), never a problem; the empty prompt of
// the "Chạy thử" step shows it as its placeholder and the step's params are never written by it; the trial result
// says the question really asked and where it came from.
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.stubEnv("VITE_MOCK_API", "1");
afterEach(() => vi.restoreAllMocks());

const hint = (node: string) => ({
  code: "MODEL-007",
  severity: "info" as const,
  node_id: node,
  param: "prompt",
  field_hint: true,
  inner_path: [],
  params: { node, prompt: "Đọc hoá đơn này" },
  title: "…",
  what: "…",
  action_ids: [],
});

describe("the trained question", () => {
  it("is out of the issue list, the prompt's placeholder, and never written into the step", async () => {
    const { api } = await import("../src/api");
    const { useWorkflow } = await import("../src/store/workflow");
    const { t } = await import("../src/i18n");
    await useWorkflow.getState().loadCatalog();
    useWorkflow.getState().newWorkflow();
    const id = useWorkflow.getState().addNode("serve.vllm", { x: 0, y: 0 }) as string;
    const real = await api.validate(useWorkflow.getState().workflow as never, "vi");
    vi.spyOn(api, "validate").mockResolvedValue({ ...real, errors: [...real.errors, hint(id)] });
    const pastBefore = useWorkflow.getState().past.length;
    await useWorkflow.getState().validate();
    const v = useWorkflow.getState().validation;
    expect(v?.errors.some((e) => e.code === "MODEL-007")).toBe(false); // nothing lists or counts it
    expect(v?.fieldHints?.map((e) => e.code)).toEqual(["MODEL-007"]);
    expect(useWorkflow.getState().past.length).toBe(pastBefore); // no edit, no undo step
    useWorkflow.getState().select(id);
    const { Inspector } = await import("../src/components/Inspector");
    render(<Inspector />);
    const box = document.getElementById("param-prompt") as HTMLTextAreaElement;
    expect(box.value).toBe("");
    expect(box.placeholder).toBe("Đọc hoá đơn này");
    expect(screen.getByText(t("trial.prefilled_prompt"))).toBeTruthy();
    fireEvent.change(box, { target: { value: "Cửa hàng tên gì?" } });
    await waitFor(() => expect(screen.queryByText(t("trial.prefilled_prompt"))).toBeNull());
    expect(useWorkflow.getState().workflow?.nodes.find((n) => n.id === id)?.params.prompt).toBe(
      "Cửa hàng tên gì?",
    );
  });

  it("the trial result says the question asked and its source; nothing for a one-task model", async () => {
    const { api } = await import("../src/api");
    const { tp, t } = await import("../src/i18n");
    const base = await api.nodeResult("r", "n");
    const spy = vi.spyOn(api, "nodeResult").mockResolvedValue({
      ...base,
      prompt_used: "Đọc hoá đơn này",
      prompt_source: "trained",
    });
    const { TrialResult } = await import("../src/components/TrialResult");
    const { container, unmount } = render(<TrialResult runId="r" nodeId="n" />);
    await waitFor(() => expect(container.querySelector(".trial-prompt")).toBeTruthy());
    const line = container.querySelector(".trial-prompt") as HTMLElement;
    expect(line.querySelector("code.mono")?.textContent).toBe("Đọc hoá đơn này");
    expect(line.textContent).toBe(
      `${tp(["trial", "prompt_used"], { prompt: "Đọc hoá đơn này" })} (${t("trial.prompt_source.trained")})`,
    );
    unmount();
    spy.mockResolvedValue({ ...base, prompt_used: "", prompt_source: "fixed" });
    const second = render(<TrialResult runId="r" nodeId="n" />);
    await waitFor(() => expect(second.container.querySelector(".trial-text")).toBeTruthy());
    expect(second.container.querySelector(".trial-prompt")).toBeNull();
  });
});
