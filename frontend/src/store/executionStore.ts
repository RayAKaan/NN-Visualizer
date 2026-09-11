import { create } from "zustand";
import { devtools } from "zustand/middleware";
import { fetchTraceDetail, runTrace } from "../api/execution";
import { useLabStore } from "./labStore";
import type {
  ConvCellDetail,
  DenseNeuronDetail,
  ExecutionCursorState,
  ExecutionTrace,
  LabViewMode,
  LstmDetail,
  Operation,
  SelectionState,
  TraceStage,
} from "../types/execution";

interface ExecutionStoreState {
  viewMode: LabViewMode;
  cursor: ExecutionCursorState;
  trace: ExecutionTrace | null;
  stageIndex: number;
  isPlaying: boolean;
  speed: number;
  isLoading: boolean;
  isDetailLoading: boolean;
  error: string | null;

  neuronDetail: DenseNeuronDetail | null;
  convDetail: ConvCellDetail | null;
  lstmDetail: LstmDetail | null;

  selection: SelectionState;
  _timer: ReturnType<typeof setInterval> | null;

  // presentation
  setViewMode: (mode: LabViewMode) => void;
  hasInput: () => boolean;

  // execution
  run: () => Promise<void>;
  step: (delta?: 1 | -1) => void;
  jumpTo: (index: number) => void;
  play: () => void;
  pause: () => void;
  reset: () => void;
  setSpeed: (speed: number) => void;

  // navigation / selection
  selectLayer: (layerId: string) => void;
  selectNeuron: (index: number) => void;
  selectFilter: (index: number) => void;
  selectPosition: (h: number, w: number) => void;
  selectTimestep: (t: number) => void;

  // internal detail loaders (used by navigation + visualizers)
  openNeuronDetail: (index: number) => Promise<void>;
  openConvDetail: (filter: number, h: number, w: number) => Promise<void>;
  openLstmDetail: (timestep: number) => Promise<void>;
  ensureStageDetail: (index: number) => void;
}

export const selectCurrentStage = (s: { trace: ExecutionTrace | null; stageIndex: number }): TraceStage | null =>
  s.trace && s.stageIndex >= 0 && s.stageIndex < s.trace.stages.length ? s.trace.stages[s.stageIndex] : null;

function defaultSelectionFor(stage: TraceStage, prev: SelectionState): SelectionState {
  const sel = { ...prev, position: prev.position ?? null };
  if (stage.operation === "conv2d") {
    const [h, w] = stage.outputShape;
    sel.position = { h: Math.max(0, Math.floor((h - 1) / 2)), w: Math.max(0, Math.floor((w - 1) / 2)) };
  }
  return sel;
}

const INITIAL: Omit<
  ExecutionStoreState,
  | "setViewMode" | "run" | "step" | "jumpTo" | "play" | "pause" | "reset" | "setSpeed"
  | "selectLayer" | "selectNeuron" | "selectFilter" | "selectPosition" | "selectTimestep" | "hasInput"
  | "openNeuronDetail" | "openConvDetail" | "openLstmDetail" | "ensureStageDetail"
> = {
  viewMode: "execution",
  cursor: "idle",
  trace: null,
  stageIndex: -1,
  isPlaying: false,
  speed: 1,
  isLoading: false,
  isDetailLoading: false,
  error: null,
  neuronDetail: null,
  convDetail: null,
  lstmDetail: null,
  selection: { neuronIndex: 0, filterIndex: 0, position: null, timestep: 0 },
  _timer: null,
};

export const useExecutionStore = create<ExecutionStoreState>()(
  devtools((set, get) => ({
    ...INITIAL,

    setViewMode: (mode) => {
      set({ viewMode: mode });
      if (mode === "classic") get().pause();
    },

    hasInput: () => useLabStore.getState().inputPixels.some((v) => v > 0.05),

    run: async () => {
      if (get().isLoading) return;
      const lab = useLabStore.getState();
      if (!lab.inputPixels.some((v) => v > 0.05)) return;
      set({ isLoading: true, cursor: "computing", error: null, isPlaying: false });
      get()._timer && clearInterval(get()._timer!);
      try {
        const trace = await runTrace({
          architecture: lab.architecture,
          dataset: lab.dataset,
          pixels: Array.from(lab.inputPixels),
        });
        set({ trace, stageIndex: 0, cursor: "completed", isLoading: false });
        void get().ensureStageDetail(0);
      } catch (err: any) {
        set({
          isLoading: false,
          cursor: "error",
          error: err?.message ?? "Execution trace failed",
        });
      }
    },

    step: (delta = 1) => {
      const { trace, stageIndex, isPlaying } = get();
      if (!trace) return;
      const next = Math.min(Math.max(stageIndex + delta, 0), trace.stages.length - 1);
      set({ stageIndex: next });
      void get().ensureStageDetail(next);
      if (next >= trace.stages.length - 1 && isPlaying) {
        set({ isPlaying: false });
        if (get()._timer) clearInterval(get()._timer!);
      }
    },

    jumpTo: (index) => {
      const { trace } = get();
      if (!trace) return;
      const next = Math.min(Math.max(index, 0), trace.stages.length - 1);
      set({ stageIndex: next });
      void get().ensureStageDetail(next);
    },

    play: () => {
      const { trace, isPlaying, speed } = get();
      if (!trace || isPlaying) return;
      if (get().stageIndex >= (trace.stages.length - 1)) {
        set({ stageIndex: 0 });
        void get().ensureStageDetail(0);
      }
      set({ isPlaying: true });
      const tick = () => {
        const { isPlaying: p, step } = get();
        if (!p) return;
        step(1);
      };
      const delay = Math.max(250, 1500 / get().speed);
      const timer = setInterval(tick, delay);
      set({ _timer: timer });
    },

    pause: () => {
      if (get()._timer) clearInterval(get()._timer!);
      set({ isPlaying: false, _timer: null });
    },

    reset: () => {
      get().pause();
      set({ trace: null, stageIndex: -1, cursor: "idle", error: null,
            neuronDetail: null, convDetail: null, lstmDetail: null });
    },

    setSpeed: (speed) => {
      set({ speed: Math.min(4, Math.max(0.5, speed)) });
      const { isPlaying, _timer } = get();
      if (isPlaying && _timer) {
        clearInterval(_timer);
        const timer = setInterval(() => get().step(1), Math.max(250, 1500 / get().speed));
        set({ _timer: timer });
      }
    },

    selectLayer: (layerId) => {
      const { trace } = get();
      if (!trace) return;
      const index = trace.stages.findIndex((s) => s.layerId === layerId || s.stageId === layerId);
      if (index >= 0) get().jumpTo(index);
    },

    selectNeuron: (index) => {
      const { selection } = get();
      set({ selection: { ...selection, neuronIndex: index } });
      void get().openNeuronDetail(index);
    },

    selectFilter: (index) => {
      const { selection } = get();
      set({ selection: { ...selection, filterIndex: index } });
      const pos = get().selection.position;
      if (pos) void get().openConvDetail(index, pos.h, pos.w);
    },

    selectPosition: (h, w) => {
      const { selection } = get();
      set({ selection: { ...selection, position: { h, w } } });
      void get().openConvDetail(selection.filterIndex, h, w);
    },

    selectTimestep: (t) => {
      const { selection } = get();
      set({ selection: { ...selection, timestep: t } });
      void get().openLstmDetail(t);
    },

    // Internal, exposed on the state object for convenience.
    openNeuronDetail: async (index: number) => {
      const lab = useLabStore.getState();
      const stage = selectCurrentStage(get());
      if (!stage) return;
      if (stage.operation !== "dense") return;
      set({ isDetailLoading: true });
      try {
        const detail = await fetchTraceDetail<DenseNeuronDetail>({
          architecture: lab.architecture,
          dataset: lab.dataset,
          pixels: Array.from(lab.inputPixels),
          layerId: stage.layerId,
          detail: "neuron",
          selection: { neuronIndex: index },
        });
        set({ neuronDetail: detail, isDetailLoading: false });
      } catch {
        set({ isDetailLoading: false });
      }
    },

    openConvDetail: async (filter: number, h: number, w: number) => {
      const lab = useLabStore.getState();
      const stage = selectCurrentStage(get());
      if (!stage) return;
      if (stage.operation !== "conv2d") return;
      set({ isDetailLoading: true });
      try {
        const detail = await fetchTraceDetail<ConvCellDetail>({
          architecture: lab.architecture,
          dataset: lab.dataset,
          pixels: Array.from(lab.inputPixels),
          layerId: stage.layerId,
          detail: "conv_cell",
          selection: { filterIndex: filter, position: { h, w } },
        });
        detail.featureMaps = stage.feature_maps;
        detail.selectedChannel = filter;
        set({ convDetail: detail, isDetailLoading: false });
      } catch {
        set({ isDetailLoading: false });
      }
    },

    openLstmDetail: async (timestep: number) => {
      const lab = useLabStore.getState();
      const stage = selectCurrentStage(get());
      if (!stage) return;
      if (!stage.operation.startsWith("lstm")) return;
      set({ isDetailLoading: true });
      try {
        const detail = await fetchTraceDetail<LstmDetail>({
          architecture: lab.architecture,
          dataset: lab.dataset,
          pixels: Array.from(lab.inputPixels),
          layerId: stage.layerId,
          detail: "lstm_gates",
          selection: { timestep },
        });
        set({ lstmDetail: detail, isDetailLoading: false });
      } catch {
        set({ isDetailLoading: false });
      }
    },

    ensureStageDetail: (index: number) => {
      const stage = get().trace?.stages[index];
      if (!stage) return;
      const { selection } = get();
      const sel = defaultSelectionFor(stage, selection);
      if (JSON.stringify(sel.position) !== JSON.stringify(selection.position)) set({ selection: sel });
      const op: Operation = stage.operation;
      if (op === "dense") {
        void get().openNeuronDetail(selectedNeuronIndex(sel));
      } else if (op === "conv2d") {
        const pos = sel.position;
        if (pos) void get().openConvDetail(sel.filterIndex, pos.h, pos.w);
      } else if (op.startsWith("lstm")) {
        void get().openLstmDetail(sel.timestep);
      }
    },
  })),
);

function selectedNeuronIndex(sel: SelectionState): number {
  return sel.neuronIndex;
}