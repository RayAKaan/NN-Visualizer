import { create } from "zustand";

interface ProfilerStore {
  isProfilerVisible: boolean;
  toggleProfiler: () => void;
  setProfilerVisible: (value: boolean) => void;
}

export const useProfilerStore = create<ProfilerStore>((set) => ({
  isProfilerVisible: false,
  toggleProfiler: () => set((s) => ({ isProfilerVisible: !s.isProfilerVisible })),
  setProfilerVisible: (value) => set({ isProfilerVisible: value }),
}));
