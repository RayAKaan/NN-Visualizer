/// <reference types="vite/client" />
import axios from "axios";

// Keep browser-facing requests relative so the app works from localhost, the
// live preview host, and an embedded deployment. In development Vite proxies
// /backend/* to FastAPI and rewrites the prefix away.
const DEV_BACKEND_PREFIX = "/backend";
const browserOrigin = typeof window !== "undefined"
  ? `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}`
  : "";

export const API_URL = import.meta.env.DEV ? DEV_BACKEND_PREFIX : "";
export const WS_URL = `${browserOrigin}${DEV_BACKEND_PREFIX}/train`;
export const SIM_WS_URL = `${browserOrigin}${DEV_BACKEND_PREFIX}/ws/simulator/train`;

export const apiClient = axios.create({
  baseURL: API_URL,
  headers: {
    "Content-Type": "application/json",
  },
});
