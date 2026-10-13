// "Chạy thử" (B-7): the image param stores only the server's upload_id; the result is shown as plain text.
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.stubEnv("VITE_MOCK_API", "1");
afterEach(() => vi.restoreAllMocks());

describe("ImageUpload", () => {
  it("uploads the file and keeps only the upload_id", async () => {
    const { ImageUpload } = await import("../src/components/ImageUpload");
    const { api } = await import("../src/api");
    const id = "0123456789abcdef0123456789abcdef";
    const spy = vi.spyOn(api, "upload").mockResolvedValue({
      upload_id: id,
      label: "hoá đơn.png",
      width: 10,
      height: 10,
      size_bytes: 4,
    });
    const onChange = vi.fn();
    render(<ImageUpload id="img" label="Ảnh" value="" onChange={onChange} />);
    const file = new File([new Uint8Array([1, 2, 3, 4])], "hoá đơn.png", { type: "image/png" });
    fireEvent.change(screen.getByLabelText("Ảnh"), { target: { files: [file] } });
    await waitFor(() => expect(onChange).toHaveBeenCalledWith(id));
    expect(spy).toHaveBeenCalledWith(file, "vi");
  });

  it("refuses a file over 20 MB before uploading and shows the server's DATA-005 text otherwise", async () => {
    const { ImageUpload } = await import("../src/components/ImageUpload");
    const { api } = await import("../src/api");
    const { ApiError } = await import("../src/api/client");
    const spy = vi.spyOn(api, "upload").mockRejectedValue(
      new ApiError(415, {
        error: { code: "DATA-005", title: "Ảnh không dùng được", what: "Không phải PNG." },
      }),
    );
    render(<ImageUpload id="img" label="Ảnh" value="" onChange={() => {}} />);
    const big = new File([new Uint8Array(1)], "x.png", { type: "image/png" });
    Object.defineProperty(big, "size", { value: 21 * 1024 * 1024 });
    fireEvent.change(screen.getByLabelText("Ảnh"), { target: { files: [big] } });
    expect(await screen.findByRole("alert")).toBeTruthy();
    expect(spy).not.toHaveBeenCalled();
    const small = new File([new Uint8Array(1)], "x.gif", { type: "image/gif" });
    fireEvent.change(screen.getByLabelText("Ảnh"), { target: { files: [small] } });
    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("Không phải PNG."));
  });
});

describe("TrialResult", () => {
  it("renders the container's text as plain text, never as markup", async () => {
    const { TrialResult } = await import("../src/components/TrialResult");
    const { api } = await import("../src/api");
    vi.spyOn(api, "nodeResult").mockResolvedValue({
      engine_used: "hf",
      text: '<img src=x onerror="alert(1)">‮total',
      n_tokens: 3,
      truncated: true,
      decoding: { note: "<b>x</b>" },
      overrides: {},
    });
    const { container } = render(<TrialResult runId="r" nodeId="n" />);
    await waitFor(() => expect(screen.getByText(/onerror/)).toBeTruthy());
    expect(container.querySelector("img[src=x]")).toBeNull();
    expect(container.textContent).toContain("\\u202E"); // bidi override shown escaped
    expect(screen.getByText(/Câu trả lời bị cắt/)).toBeTruthy();
  });

  it("explains RESULT_TOO_LARGE instead of failing", async () => {
    const { TrialResult } = await import("../src/components/TrialResult");
    const { api } = await import("../src/api");
    const { ApiError } = await import("../src/api/client");
    vi.spyOn(api, "nodeResult").mockRejectedValue(
      new ApiError(502, { error: { code: "RESULT_TOO_LARGE" } }),
    );
    render(<TrialResult runId="r" nodeId="n" />);
    expect(await screen.findByText("Kết quả quá dài nên không hiển thị được ở đây.")).toBeTruthy();
  });
});
