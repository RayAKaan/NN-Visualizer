import { create } from "zustand";
import { simulatorApi } from "../hooks/useSimulatorApi";

interface AssistantMessage {
  role: "user" | "assistant";
  text: string;
}

interface AssistantState {
  open: boolean;
  loading: boolean;
  messages: AssistantMessage[];
  toggle: () => void;
  send: (text: string) => Promise<void>;
}

export const useAssistantStore = create<AssistantState>((set, get) => ({
  open: false,
  loading: false,
  messages: [],
  toggle: () => set((s) => ({ open: !s.open })),
  async send(text) {
    set((s) => ({ loading: true, messages: [...s.messages, { role: "user", text }] }));
    const res = await simulatorApi.assistantQuery(text, {});
    set((s) => ({ loading: false, messages: [...s.messages, { role: "assistant", text: res.reply }] }));
  },
}));
