import { type Api, httpApi } from "./client";
import { mockApi } from "./mock";

export const api: Api = import.meta.env.VITE_MOCK_API === "1" ? mockApi : httpApi;
export * from "./types";
