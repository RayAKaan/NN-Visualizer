import { create } from "zustand";
import { devtools } from "zustand/middleware";
import { simulatorApi } from "../hooks/useSimulatorApi";
import { useSimulatorStore } from "./simulatorStore";

interface ReplayState {
  snapshots: any[];
  totalSnapshots: number;
  currentSnapshotIndex: number;
  loadedEpoch: number | null;
  isPlaying: boolean;
  playbackSpeed: number;
  loadSnapshots: () => Promise<void>;
  loadSnapshot: (index: number) => Promise<void>;
  play: () => void;
  pause: () => void;
  setSpeed: (speed: number) => void;
}

export const useReplayStore = create<ReplayState>()(
  devtools((set, get) => ({
    snapshots: [],
    totalSnapshots: 0,
    currentSnapshotIndex: 0,
    loadedEpoch: null,
    isPlaying: false,
    playbackSpeed: 2,
    async loadSnapshots() {
      const graphId = useSimulatorStore.getState().graphId;
      if (!graphId) return;
      const res = await simulatorApi.replaySnapshots(graphId);
      set({ snapshots: res.summaries ?? [], totalSnapshots: res.total_snapshots ?? 0 });
    },
    async loadSnapshot(index) {
      const graphId = useSimulatorStore.getState().graphId;
      if (!graphId) return;
      const res = await simulatorApi.replayLoad(graphId, index);
      set({ currentSnapshotIndex: index, loadedEpoch: res.epoch });
    },
    play() {
      set({ isPlaying: true });
    },
    pause() {
      set({ isPlaying: false });
    },
    setSpeed(speed) {
      set({ playbackSpeed: speed });
    },
  })),
);

