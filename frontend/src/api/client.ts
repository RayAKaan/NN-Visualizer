import axios from "axios";

export const API_URL = "http://localhost:8000";
export const WS_URL = "ws://localhost:8000/train";
export const SIM_WS_URL = "ws://localhost:8000/ws/simulator/train";

export const apiClient = axios.create({
  baseURL: API_URL,
  headers: {
    "Content-Type": "application/json",
  },
});
