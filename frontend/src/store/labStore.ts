import { create } from "zustand";
import { devtools } from "zustand/middleware";
import { apiClient } from "../api/client";
import { buildPipelineStages } from "../data/equations";
import type {
  Architecture,
  BackwardStageActivation,
  Dataset,
  EpochCheckpoint,
  LossInfo,
  ManipulationSnapshot,
  PipelineState,
  SaliencyData,
  StageActivation,
  StageStatus,
  WeightInspectionData,
  WeightManipulation,
} from "../types/pipeline";

interface LabStore extends PipelineState {
  inputPixels: Float32Array;
  inputImageUrl: string | null;
  autoTimer: ReturnType<typeof setTimeout> | null;
  _replayTimer: ReturnType<typeof setInterval> | null;
  _debouncedReInference: ReturnType<typeof setTimeout> | null;
  _reInferenceAbort: AbortController | null;

  setArchitecture: (arch: Architecture) => void;
  setDataset: (ds: Dataset) => void;
  setSpeed: (speed: number) => void;
  setLearningRate: (value: number) => void;

  setInputPixels: (pixels: Float32Array) => void;
  setInputImage: (imageUrl: string, pixels: Float32Array) => void;

  startPipeline: () => Promise<void>;
  stepForward: () => Promise<void>;
  stepBackward: () => void;
  skipToEnd: () => Promise<void>;
  resetPipeline: () => void;
  pausePipeline: () => void;
  resumePipeline: () => void;

  setTrueLabel: (label: number) => void;
  startBackwardPass: () => Promise<void>;
  stepBackward_bwd: () => Promise<void>;
  stepForward_bwd: () => void;
  skipToEnd_bwd: () => Promise<void>;
  resetBackwardPass: () => void;

  setComparisonMode: (mode: "off" | "trained" | "untrained") => void;
  fetchUntrainedActivation: (stageIndex: number) => Promise<void>;

  computeSaliency: () => Promise<void>;

  inspectStageWeights: (stageId: string) => Promise<void>;
  clearWeightInspection: () => void;

  enableManipulationMode: () => void;
  disableManipulationMode: () => void;
  setWeight: (stageId: string, flatIndex: number, value: number) => void;
  resetWeight: (stageId: string, flatIndex: number) => void;
  resetAllWeights: (stageId: string) => void;
  resetAllManipulations: () => void;
  knockoutNeuron: (stageId: string, neuronIndex: number) => void;
  restoreNeuron: (stageId: string, neuronIndex: number) => void;
  bypassLayer: (stageId: string) => void;
  restoreLayer: (stageId: string) => void;
  randomizeWeights: (stageId: string, scale: number) => void;
  triggerReInference: () => Promise<void>;

  enableReplayMode: () => Promise<void>;
  disableReplayMode: () => void;
  scrubToEpoch: (epoch: number) => Promise<void>;
  replayStepForward: () => void;
  replayStepBackward: () => void;
  startReplayAutoPlay: () => void;
  stopReplayAutoPlay: () => void;
  setReplaySpeed: (speed: number) => void;
  fetchEpochCheckpoint: (epoch: number) => Promise<EpochCheckpoint>;
  preloadAdjacentEpochs: (currentEpoch: number) => void;

  fetchStageActivation: (stageIndex: number, useUntrained?: boolean) => Promise<StageActivation>;
  advanceStage: (stageIndex: number) => Promise<void>;

  fetchLoss: () => Promise<LossInfo>;
  fetchBackwardStageActivation: (stageIndex: number) => Promise<BackwardStageActivation>;
  advanceBackwardStage: (stageIndex: number) => Promise<void>;
  fetchSaliency: () => Promise<SaliencyData>;
  fetchWeightData: (stageId: string) => Promise<WeightInspectionData>;
}

function lockedStatuses(ids: string[]): Record<string, StageStatus> {
  return ids.reduce<Record<string, StageStatus>>((acc, id) => {
    acc[id] = "locked";
    return acc;
  }, {});
}

function parseStageActivation(data: any, stageId: string): StageActivation {
  return {
    stageId,
    inputData: new Float32Array(data.input ?? []),
    outputData: new Float32Array(data.output ?? []),
    weights: data.weights ? new Float32Array(data.weights) : undefined,
    bias: data.bias ? new Float32Array(data.bias) : undefined,
    kernels: Array.isArray(data.kernels) ? data.kernels.map((k: number[]) => new Float32Array(k)) : undefined,
    gateValues: data.gates
      ? {
          forget: new Float32Array(data.gates.forget ?? []),
          input: new Float32Array(data.gates.input ?? []),
          output: new Float32Array(data.gates.output ?? []),
          cellState: new Float32Array(data.gates.cell_state ?? []),
        }
      : undefined,
    metadata: {
      inputShape: Array.isArray(data.input_shape) ? data.input_shape : [],
      outputShape: Array.isArray(data.output_shape) ? data.output_shape : [],
      paramCount: typeof data.param_count === "number" ? data.param_count : 0,
      computeTimeMs: typeof data.compute_time_ms === "number" ? data.compute_time_ms : 0,
    },
  };
}

function parseEpochCheckpoint(data: any): EpochCheckpoint {
  const layerSnapshots: EpochCheckpoint["layerSnapshots"] = {};
  const raw = data.layerSnapshots ?? {};
  Object.entries(raw).forEach(([stageId, entry]: [string, any]) => {
    layerSnapshots[stageId] = {
      stageId,
      weights: entry.weights ? new Float32Array(entry.weights) : undefined,
      bias: entry.bias ? new Float32Array(entry.bias) : undefined,
      activationSample: entry.activationSample ? new Float32Array(entry.activationSample) : undefined,
      featureMapSample: Array.isArray(entry.featureMapSample)
        ? entry.featureMapSample.map((v: number[]) => new Float32Array(v))
        : undefined,
      statistics: {
        weightMean: entry.statistics?.weightMean ?? 0,
        weightStd: entry.statistics?.weightStd ?? 0,
        activationMean: entry.statistics?.activationMean ?? 0,
        activationSparsity: entry.statistics?.activationSparsity ?? 0,
        gradientNorm: entry.statistics?.gradientNorm ?? 0,
      },
    };
  });
  return {
    epoch: data.epoch ?? 0,
    metrics: {
      trainLoss: data.metrics?.trainLoss ?? 0,
      valLoss: data.metrics?.valLoss ?? 0,
      trainAccuracy: data.metrics?.trainAccuracy ?? 0,
      valAccuracy: data.metrics?.valAccuracy ?? 0,
      learningRate: data.metrics?.learningRate ?? 0,
    },
    layerSnapshots,
  };
}

export const useLabStore = create<LabStore>()(
  devtools(
    (set, get) => {
      const initialStages = buildPipelineStages("ANN", "mnist");
      return {
        architecture: "ANN",
        dataset: "mnist",
        stages: initialStages,
        currentStageIndex: -1,
        stageStatuses: lockedStatuses(initialStages.map((s) => s.id)),
        activations: {},
        finalPrediction: null,
        isRunning: false,
        speed: 1,
        inputPixels: new Float32Array(28 * 28),
        inputImageUrl: null,
        autoTimer: null,
        _replayTimer: null,
        _debouncedReInference: null,
        _reInferenceAbort: null,

        passDirection: "forward",
        trueLabel: null,
        lossInfo: null,
        backwardActivations: {},
        backwardStageStatuses: {},
        currentBackwardStageIndex: -1,
        saliencyData: null,
        comparisonMode: "off",
        untrainedActivations: {},
        inspectedStageId: null,
        weightInspection: null,
        learningRate: 0.001,

        isManipulationMode: false,
        activeManipulations: {},
        manipulationHistory: [],
        bypassedLayers: new Set(),
        knockedOutNeurons: {},
        modifiedWeights: {},
        manipulatedActivations: {},
        manipulatedPrediction: null,
        isReInferencing: false,

        isReplayMode: false,
        availableEpochs: [],
        currentReplayEpoch: 0,
        epochCheckpoints: {},
        isLoadingEpoch: false,
        replayAutoPlaySpeed: 1,
        isReplayAutoPlaying: false,

        setArchitecture(architecture) {
          const stages = buildPipelineStages(architecture, get().dataset);
          set({
            architecture,
            stages,
            stageStatuses: lockedStatuses(stages.map((s) => s.id)),
            currentStageIndex: -1,
            activations: {},
            finalPrediction: null,
            isRunning: false,
            passDirection: "forward",
            lossInfo: null,
            backwardActivations: {},
            backwardStageStatuses: {},
            currentBackwardStageIndex: -1,
            saliencyData: null,
            comparisonMode: "off",
            untrainedActivations: {},
            isManipulationMode: false,
            activeManipulations: {},
            manipulationHistory: [],
            bypassedLayers: new Set(),
            knockedOutNeurons: {},
            modifiedWeights: {},
            manipulatedActivations: {},
            manipulatedPrediction: null,
            isReInferencing: false,
            isReplayMode: false,
            availableEpochs: [],
            currentReplayEpoch: 0,
            epochCheckpoints: {},
            isLoadingEpoch: false,
            replayAutoPlaySpeed: 1,
            isReplayAutoPlaying: false,
            _replayTimer: null,
          });
        },

        setDataset(dataset) {
          const stages = buildPipelineStages(get().architecture, dataset);
          set({
            dataset,
            stages,
            stageStatuses: lockedStatuses(stages.map((s) => s.id)),
            currentStageIndex: -1,
            activations: {},
            finalPrediction: null,
            isRunning: false,
            inputImageUrl: null,
            passDirection: "forward",
            lossInfo: null,
            backwardActivations: {},
            backwardStageStatuses: {},
            currentBackwardStageIndex: -1,
            saliencyData: null,
            comparisonMode: "off",
            untrainedActivations: {},
            isManipulationMode: false,
            activeManipulations: {},
            manipulationHistory: [],
            bypassedLayers: new Set(),
            knockedOutNeurons: {},
            modifiedWeights: {},
            manipulatedActivations: {},
            manipulatedPrediction: null,
            isReInferencing: false,
            isReplayMode: false,
            availableEpochs: [],
            currentReplayEpoch: 0,
            epochCheckpoints: {},
            isLoadingEpoch: false,
            replayAutoPlaySpeed: 1,
            isReplayAutoPlaying: false,
            _replayTimer: null,
          });
        },

        setSpeed(speed) {
          set({ speed });
        },

        setLearningRate(learningRate) {
          set({ learningRate });
        },

        setInputPixels(inputPixels) {
          set({ inputPixels, inputImageUrl: null });
          get().resetPipeline();
        },

        setInputImage(inputImageUrl, inputPixels) {
          set({ inputImageUrl, inputPixels });
          get().resetPipeline();
        },

        async startPipeline() {
          const stages = get().stages;
          set({
            currentStageIndex: -1,
            stageStatuses: lockedStatuses(stages.map((s) => s.id)),
            activations: {},
            finalPrediction: null,
            isRunning: true,
            passDirection: "forward",
            lossInfo: null,
            backwardActivations: {},
            backwardStageStatuses: {},
            currentBackwardStageIndex: -1,
            saliencyData: null,
          });
          await get().advanceStage(0);
        },

        async stepForward() {
          const nextIndex = get().currentStageIndex + 1;
          if (nextIndex >= get().stages.length) return;
          if (!get().isRunning) set({ isRunning: true });
          await get().advanceStage(nextIndex);
          set({ isRunning: false });
        },

        stepBackward() {
          const { currentStageIndex, stages, stageStatuses } = get();
          if (currentStageIndex <= 0) return;
          const prevIndex = currentStageIndex - 1;
          const nextStatuses = { ...stageStatuses };
          nextStatuses[stages[currentStageIndex].id] = "locked";
          nextStatuses[stages[prevIndex].id] = "active";
          set({ currentStageIndex: prevIndex, stageStatuses: nextStatuses, finalPrediction: null });
        },

        async skipToEnd() {
          set({ isRunning: true });
          for (let i = get().currentStageIndex + 1; i < get().stages.length; i += 1) {
            await get().advanceStage(i);
          }
        },

        resetPipeline() {
          const timer = get().autoTimer;
          if (timer) clearTimeout(timer);
          const replayTimer = get()._replayTimer;
          if (replayTimer) clearInterval(replayTimer);
          const debounceTimer = get()._debouncedReInference;
          if (debounceTimer) clearTimeout(debounceTimer);
          get()._reInferenceAbort?.abort();
          const stages = get().stages;
          set({
            currentStageIndex: -1,
            stageStatuses: lockedStatuses(stages.map((s) => s.id)),
            activations: {},
            finalPrediction: null,
            isRunning: false,
            autoTimer: null,
            passDirection: "forward",
            trueLabel: null,
            lossInfo: null,
            backwardActivations: {},
            backwardStageStatuses: {},
            currentBackwardStageIndex: -1,
            saliencyData: null,
            comparisonMode: "off",
            untrainedActivations: {},
            inspectedStageId: null,
            weightInspection: null,
            isManipulationMode: false,
            activeManipulations: {},
            manipulationHistory: [],
            bypassedLayers: new Set(),
            knockedOutNeurons: {},
            modifiedWeights: {},
            manipulatedActivations: {},
            manipulatedPrediction: null,
            isReInferencing: false,
            _debouncedReInference: null,
            _reInferenceAbort: null,
            isReplayMode: false,
            availableEpochs: [],
            currentReplayEpoch: 0,
            epochCheckpoints: {},
            isLoadingEpoch: false,
            replayAutoPlaySpeed: 1,
            isReplayAutoPlaying: false,
            _replayTimer: null,
          });
        },

        pausePipeline() {
          const timer = get().autoTimer;
          if (timer) clearTimeout(timer);
          set({ isRunning: false, autoTimer: null });
        },

        resumePipeline() {
          set({ isRunning: true });
          const next = get().currentStageIndex + 1;
          if (next < get().stages.length) void get().advanceStage(next);
        },

        setTrueLabel(trueLabel) {
          set({ trueLabel });
        },

        async startBackwardPass() {
          if (get().trueLabel === null || !get().finalPrediction) return;
          const lossInfo = await get().fetchLoss();
          const trainable = get().stages.filter((s) => !["input", "output"].includes(s.type));
          set({
            passDirection: "backward",
            lossInfo,
            backwardStageStatuses: lockedStatuses(trainable.map((s) => s.id)),
            backwardActivations: {},
            currentBackwardStageIndex: -1,
            isRunning: true,
          });
          await get().advanceBackwardStage(0);
        },

        async stepBackward_bwd() {
          const trainable = get().stages.filter((s) => !["input", "output"].includes(s.type));
          const next = get().currentBackwardStageIndex + 1;
          if (next >= trainable.length) return;
          if (!get().isRunning) set({ isRunning: true });
          await get().advanceBackwardStage(next);
          set({ isRunning: false });
        },

        stepForward_bwd() {
          const trainable = get().stages.filter((s) => !["input", "output"].includes(s.type)).reverse();
          const current = get().currentBackwardStageIndex;
          if (current <= 0) return;
          const prev = current - 1;
          const statuses = { ...get().backwardStageStatuses };
          statuses[trainable[current].id] = "locked";
          statuses[trainable[prev].id] = "active";
          set({ currentBackwardStageIndex: prev, backwardStageStatuses: statuses });
        },

        async skipToEnd_bwd() {
          const trainable = get().stages.filter((s) => !["input", "output"].includes(s.type));
          set({ isRunning: true });
          for (let i = get().currentBackwardStageIndex + 1; i < trainable.length; i += 1) {
            await get().advanceBackwardStage(i);
          }
        },

        resetBackwardPass() {
          set({
            passDirection: "forward",
            lossInfo: null,
            backwardActivations: {},
            backwardStageStatuses: {},
            currentBackwardStageIndex: -1,
            saliencyData: null,
            isRunning: false,
          });
        },

        setComparisonMode(comparisonMode) {
          set({ comparisonMode });
          if (comparisonMode === "untrained") {
            const { stages, stageStatuses } = get();
            stages.forEach((stage, index) => {
              if (stageStatuses[stage.id] === "completed" || stageStatuses[stage.id] === "active") {
                void get().fetchUntrainedActivation(index);
              }
            });
          }
        },

        async fetchUntrainedActivation(stageIndex) {
          const stage = get().stages[stageIndex];
          if (!stage) return;
          const activation = await get().fetchStageActivation(stageIndex, true);
          set((s) => ({ untrainedActivations: { ...s.untrainedActivations, [stage.id]: activation } }));
        },

        async computeSaliency() {
          const saliencyData = await get().fetchSaliency();
          set({ saliencyData });
        },

        async inspectStageWeights(stageId) {
          set({ inspectedStageId: stageId });
          const weightInspection = await get().fetchWeightData(stageId);
          set({ weightInspection });
        },

        clearWeightInspection() {
          set({ inspectedStageId: null, weightInspection: null });
        },

        enableManipulationMode() {
          set({
            isManipulationMode: true,
            modifiedWeights: {},
            bypassedLayers: new Set(),
            knockedOutNeurons: {},
            manipulatedPrediction: null,
            manipulatedActivations: {},
            activeManipulations: {},
          });
        },

        disableManipulationMode() {
          const timer = get()._debouncedReInference;
          if (timer) clearTimeout(timer);
          get()._reInferenceAbort?.abort();
          set({
            isManipulationMode: false,
            modifiedWeights: {},
            bypassedLayers: new Set(),
            knockedOutNeurons: {},
            manipulatedPrediction: null,
            manipulatedActivations: {},
            activeManipulations: {},
            isReInferencing: false,
            _debouncedReInference: null,
            _reInferenceAbort: null,
          });
        },

        setWeight(stageId, flatIndex, value) {
          set((s) => {
            const stageWeights = new Map(s.modifiedWeights[stageId] ?? []);
            stageWeights.set(flatIndex, value);
            return { modifiedWeights: { ...s.modifiedWeights, [stageId]: stageWeights } };
          });
          const t = get()._debouncedReInference;
          if (t) clearTimeout(t);
          const timer = setTimeout(() => void get().triggerReInference(), 250);
          set({ _debouncedReInference: timer });
        },

        resetWeight(stageId, flatIndex) {
          set((s) => {
            const stageWeights = new Map(s.modifiedWeights[stageId] ?? []);
            stageWeights.delete(flatIndex);
            const modifiedWeights = { ...s.modifiedWeights };
            if (stageWeights.size === 0) delete modifiedWeights[stageId];
            else modifiedWeights[stageId] = stageWeights;
            return { modifiedWeights };
          });
          void get().triggerReInference();
        },

        resetAllWeights(stageId) {
          set((s) => {
            const modifiedWeights = { ...s.modifiedWeights };
            delete modifiedWeights[stageId];
            return { modifiedWeights };
          });
          void get().triggerReInference();
        },

        resetAllManipulations() {
          set({
            modifiedWeights: {},
            bypassedLayers: new Set(),
            knockedOutNeurons: {},
            manipulatedPrediction: null,
            manipulatedActivations: {},
            activeManipulations: {},
          });
          void get().triggerReInference();
        },

        knockoutNeuron(stageId, neuronIndex) {
          set((s) => {
            const stageSet = new Set(s.knockedOutNeurons[stageId] ?? []);
            stageSet.add(neuronIndex);
            return { knockedOutNeurons: { ...s.knockedOutNeurons, [stageId]: stageSet } };
          });
          void get().triggerReInference();
        },

        restoreNeuron(stageId, neuronIndex) {
          set((s) => {
            const stageSet = new Set(s.knockedOutNeurons[stageId] ?? []);
            stageSet.delete(neuronIndex);
            return { knockedOutNeurons: { ...s.knockedOutNeurons, [stageId]: stageSet } };
          });
          void get().triggerReInference();
        },

        bypassLayer(stageId) {
          set((s) => {
            const bypassedLayers = new Set(s.bypassedLayers);
            bypassedLayers.add(stageId);
            return { bypassedLayers };
          });
          void get().triggerReInference();
        },

        restoreLayer(stageId) {
          set((s) => {
            const bypassedLayers = new Set(s.bypassedLayers);
            bypassedLayers.delete(stageId);
            return { bypassedLayers };
          });
          void get().triggerReInference();
        },

        randomizeWeights(stageId, scale) {
          const activation = get().activations[stageId];
          if (!activation?.weights) return;
          const stageWeights = new Map<number, number>();
          for (let i = 0; i < activation.weights.length; i += 1) {
            stageWeights.set(i, (Math.random() * 2 - 1) * scale);
          }
          set((s) => ({ modifiedWeights: { ...s.modifiedWeights, [stageId]: stageWeights } }));
          void get().triggerReInference();
        },

        async triggerReInference() {
          const {
            inputPixels,
            architecture,
            dataset,
            modifiedWeights,
            bypassedLayers,
            knockedOutNeurons,
            finalPrediction,
          } = get();

          get()._reInferenceAbort?.abort();
          const controller = new AbortController();
          set({ isReInferencing: true, _reInferenceAbort: controller });

          try {
            const serializedModifications: Record<string, Record<string, number>> = {};
            Object.entries(modifiedWeights).forEach(([stageId, weightMap]) => {
              serializedModifications[stageId] = Object.fromEntries(weightMap);
            });
            const serializedKnockouts: Record<string, number[]> = {};
            Object.entries(knockedOutNeurons).forEach(([stageId, setForStage]) => {
              serializedKnockouts[stageId] = Array.from(setForStage);
            });

            const response = await apiClient.post(
              "/api/lab/manipulated-inference",
              {
                architecture,
                dataset,
                pixels: Array.from(inputPixels),
                weightModifications: serializedModifications,
                bypassedLayers: Array.from(bypassedLayers),
                knockedOutNeurons: serializedKnockouts,
              },
              { signal: controller.signal },
            );

            const data = response.data;
            const manipulatedActivations: Record<string, StageActivation> = {};
            Object.entries(data.activations ?? {}).forEach(([stageId, actData]: [string, any]) => {
              manipulatedActivations[stageId] = {
                stageId,
                inputData: new Float32Array(actData.input ?? []),
                outputData: new Float32Array(actData.output ?? []),
                metadata: {
                  inputShape: actData.metadata?.inputShape ?? [],
                  outputShape: actData.metadata?.outputShape ?? [],
                  paramCount: actData.metadata?.paramCount ?? 0,
                  computeTimeMs: actData.metadata?.computeTimeMs ?? 0,
                },
              };
            });

            const manipulatedPrediction = {
              label: data.prediction?.label,
              confidence: data.prediction?.confidence,
              probs: Array.isArray(data.prediction?.probs) ? data.prediction.probs : [],
            };

            const activeManipulations: Record<string, WeightManipulation> = {};
            Object.entries(modifiedWeights).forEach(([stageId, weightMap]) => {
              const baseWeights = get().activations[stageId]?.weights;
              const changes = Array.from(weightMap.entries()).map(([index, newValue]) => ({
                index,
                row: 0,
                col: index,
                originalValue: baseWeights?.[index] ?? 0,
                newValue,
              }));
              const delta = finalPrediction
                ? Math.abs((finalPrediction.confidence ?? 0) - (manipulatedPrediction.confidence ?? 0)) / 100
                : 0;
              const labelShift = finalPrediction && finalPrediction.label !== manipulatedPrediction.label ? 0.5 : 0;
              activeManipulations[stageId] = {
                stageId,
                type: "single_weight",
                changes,
                originalPrediction: {
                  label: finalPrediction?.label ?? "",
                  confidence: finalPrediction?.confidence ?? 0,
                },
                manipulatedPrediction: {
                  label: manipulatedPrediction.label,
                  confidence: manipulatedPrediction.confidence,
                },
                impactScore: Math.min(1, delta + labelShift),
              };
            });

            set((s) => ({
              manipulatedActivations,
              manipulatedPrediction,
              activeManipulations,
              isReInferencing: false,
              _reInferenceAbort: null,
              manipulationHistory: [
                ...s.manipulationHistory,
                {
                  timestamp: Date.now(),
                  stageId: Object.keys(modifiedWeights)[0] ?? "",
                  changes: [],
                  resultingPrediction: {
                    label: manipulatedPrediction.label,
                    confidence: manipulatedPrediction.confidence,
                  },
                } as ManipulationSnapshot,
              ].slice(-50),
            }));
          } catch (error: any) {
            if (error?.name === "CanceledError" || error?.name === "AbortError") return;
            set({ isReInferencing: false, _reInferenceAbort: null });
          }
        },

        async enableReplayMode() {
          set({ isReplayMode: true, isLoadingEpoch: true });
          try {
            const response = await apiClient.get(
              `/api/lab/replay/epochs?arch=${get().architecture}&dataset=${get().dataset}`,
            );
            const availableEpochs = Array.isArray(response.data?.epochs)
              ? response.data.epochs
              : [0, 1, 5, 10, 20, 50, 100];
            set({
              availableEpochs,
              currentReplayEpoch: availableEpochs[0] ?? 0,
              isLoadingEpoch: false,
            });
            if (availableEpochs.length > 0) await get().scrubToEpoch(availableEpochs[0]);
          } catch {
            set({ isReplayMode: false, isLoadingEpoch: false });
          }
        },

        disableReplayMode() {
          const timer = get()._replayTimer;
          if (timer) clearInterval(timer);
          set({
            isReplayMode: false,
            availableEpochs: [],
            currentReplayEpoch: 0,
            epochCheckpoints: {},
            isLoadingEpoch: false,
            isReplayAutoPlaying: false,
            _replayTimer: null,
          });
        },

        async scrubToEpoch(epoch) {
          const cached = get().epochCheckpoints[epoch];
          set({ isLoadingEpoch: true, currentReplayEpoch: epoch });
          let checkpoint = cached;
          if (!checkpoint) {
            checkpoint = await get().fetchEpochCheckpoint(epoch);
            set((s) => {
              const epochCheckpoints = { ...s.epochCheckpoints, [epoch]: checkpoint };
              const keys = Object.keys(epochCheckpoints).map(Number).sort((a, b) => a - b);
              if (keys.length > 20) delete epochCheckpoints[keys[0]];
              return { epochCheckpoints };
            });
          }
          set({ isLoadingEpoch: false });
          get().preloadAdjacentEpochs(epoch);
        },

        replayStepForward() {
          const { availableEpochs, currentReplayEpoch } = get();
          const index = availableEpochs.indexOf(currentReplayEpoch);
          if (index >= 0 && index < availableEpochs.length - 1) {
            void get().scrubToEpoch(availableEpochs[index + 1]);
          }
        },

        replayStepBackward() {
          const { availableEpochs, currentReplayEpoch } = get();
          const index = availableEpochs.indexOf(currentReplayEpoch);
          if (index > 0) {
            void get().scrubToEpoch(availableEpochs[index - 1]);
          }
        },

        startReplayAutoPlay() {
          const existing = get()._replayTimer;
          if (existing) clearInterval(existing);
          const timer = setInterval(() => {
            const { availableEpochs, currentReplayEpoch, isLoadingEpoch } = get();
            if (isLoadingEpoch) return;
            const index = availableEpochs.indexOf(currentReplayEpoch);
            if (index >= 0 && index < availableEpochs.length - 1) {
              void get().scrubToEpoch(availableEpochs[index + 1]);
            } else {
              get().stopReplayAutoPlay();
            }
          }, 1000 / Math.max(0.5, get().replayAutoPlaySpeed));
          set({ isReplayAutoPlaying: true, _replayTimer: timer });
        },

        stopReplayAutoPlay() {
          const timer = get()._replayTimer;
          if (timer) clearInterval(timer);
          set({ isReplayAutoPlaying: false, _replayTimer: null });
        },

        setReplaySpeed(replayAutoPlaySpeed) {
          set({ replayAutoPlaySpeed });
          if (get().isReplayAutoPlaying) {
            get().stopReplayAutoPlay();
            get().startReplayAutoPlay();
          }
        },

        async fetchEpochCheckpoint(epoch) {
          const response = await apiClient.post("/api/lab/replay/checkpoint", {
            architecture: get().architecture,
            dataset: get().dataset,
            epoch,
            pixels: Array.from(get().inputPixels),
          });
          return parseEpochCheckpoint(response.data);
        },

        preloadAdjacentEpochs(currentEpoch) {
          const { availableEpochs, epochCheckpoints } = get();
          const idx = availableEpochs.indexOf(currentEpoch);
          const toPreload = [availableEpochs[idx - 1], availableEpochs[idx + 1], availableEpochs[idx + 2]].filter(
            (epoch): epoch is number => typeof epoch === "number" && !epochCheckpoints[epoch],
          );
          toPreload.forEach((epoch) => {
            void get().fetchEpochCheckpoint(epoch).then((checkpoint) => {
              set((s) => ({ epochCheckpoints: { ...s.epochCheckpoints, [epoch]: checkpoint } }));
            });
          });
        },

        async fetchStageActivation(stageIndex, useUntrained = false) {
          const stage = get().stages[stageIndex];
          const response = await apiClient.post("/api/lab/activate", {
            architecture: get().architecture,
            dataset: get().dataset,
            stageId: stage.id,
            stageIndex,
            pixels: Array.from(get().inputPixels),
            useUntrainedWeights: useUntrained,
          });
          return parseStageActivation(response.data, stage.id);
        },

        async advanceStage(stageIndex) {
          const { stages, stageStatuses, speed, isRunning } = get();
          if (stageIndex >= stages.length) return;
          const stage = stages[stageIndex];
          const statuses = { ...stageStatuses };
          for (let i = 0; i < stageIndex; i += 1) statuses[stages[i].id] = "completed";
          statuses[stage.id] = "processing";
          set({ currentStageIndex: stageIndex, stageStatuses: statuses });

          try {
            const activation = await get().fetchStageActivation(stageIndex, false);
            set((state) => ({
              activations: { ...state.activations, [stage.id]: activation },
              stageStatuses: { ...state.stageStatuses, [stage.id]: "active" },
            }));

            if (stage.type === "output") {
              const probs = Array.from(activation.outputData);
              const labelIndex = probs.reduce((best, value, idx, arr) => (value > arr[best] ? idx : best), 0);
              set({
                finalPrediction: {
                  label: get().dataset === "catdog" ? (labelIndex === 0 ? "Cat" : "Dog") : labelIndex,
                  confidence: (probs[labelIndex] ?? 0) * 100,
                  probs,
                },
                isRunning: false,
                autoTimer: null,
              });
              return;
            }

            if (isRunning) {
              const delay = Math.max(200, 1500 / Math.max(0.5, speed));
              const timer = setTimeout(() => void get().advanceStage(stageIndex + 1), delay);
              set({ autoTimer: timer });
            }
          } catch {
            set((state) => ({
              stageStatuses: { ...state.stageStatuses, [stage.id]: "locked" },
              isRunning: false,
              autoTimer: null,
            }));
          }
        },

        async fetchLoss() {
          const response = await apiClient.post("/api/lab/loss", {
            architecture: get().architecture,
            dataset: get().dataset,
            pixels: Array.from(get().inputPixels),
            trueLabel: get().trueLabel,
          });
          return response.data as LossInfo;
        },

        async fetchBackwardStageActivation(stageIndex) {
          const trainable = get().stages.filter((s) => !["input", "output"].includes(s.type)).reverse();
          const stage = trainable[stageIndex];
          const response = await apiClient.post("/api/lab/backward", {
            architecture: get().architecture,
            dataset: get().dataset,
            stageId: stage.id,
            stageIndex,
            pixels: Array.from(get().inputPixels),
            trueLabel: get().trueLabel,
            learningRate: get().learningRate,
          });
          const data = response.data;
          return {
            stageId: stage.id,
            inputGradient: new Float32Array(data.input_gradient ?? []),
            outputGradient: new Float32Array(data.output_gradient ?? []),
            weightGradient: data.weight_gradient ? new Float32Array(data.weight_gradient) : undefined,
            biasGradient: data.bias_gradient ? new Float32Array(data.bias_gradient) : undefined,
            kernelGradients: Array.isArray(data.kernel_gradients)
              ? data.kernel_gradients.map((k: number[]) => new Float32Array(k))
              : undefined,
            gateGradients: data.gate_gradients
              ? {
                  forget: new Float32Array(data.gate_gradients.forget ?? []),
                  input: new Float32Array(data.gate_gradients.input ?? []),
                  output: new Float32Array(data.gate_gradients.output ?? []),
                  cellState: new Float32Array(data.gate_gradients.cell_state ?? []),
                }
              : undefined,
            stats: data.stats,
            proposedWeightDelta: data.proposed_weight_delta ? new Float32Array(data.proposed_weight_delta) : undefined,
            proposedBiasDelta: data.proposed_bias_delta ? new Float32Array(data.proposed_bias_delta) : undefined,
            metadata: {
              inputShape: Array.isArray(data.input_shape) ? data.input_shape : [],
              outputShape: Array.isArray(data.output_shape) ? data.output_shape : [],
              computeTimeMs: typeof data.compute_time_ms === "number" ? data.compute_time_ms : 0,
            },
          } as BackwardStageActivation;
        },

        async advanceBackwardStage(stageIndex) {
          const trainable = get().stages.filter((s) => !["input", "output"].includes(s.type)).reverse();
          if (stageIndex >= trainable.length || !get().isRunning) return;

          const stage = trainable[stageIndex];
          const statuses = { ...get().backwardStageStatuses };
          for (let i = 0; i < stageIndex; i += 1) statuses[trainable[i].id] = "completed";
          statuses[stage.id] = "processing";
          set({ currentBackwardStageIndex: stageIndex, backwardStageStatuses: statuses });

          try {
            const activation = await get().fetchBackwardStageActivation(stageIndex);
            set((s) => ({
              backwardActivations: { ...s.backwardActivations, [stage.id]: activation },
              backwardStageStatuses: { ...s.backwardStageStatuses, [stage.id]: "active" },
            }));

            if (stageIndex === trainable.length - 1) {
              await get().computeSaliency();
              set({ isRunning: false, autoTimer: null });
              return;
            }

            if (get().isRunning) {
              const delay = Math.max(200, 1500 / Math.max(0.5, get().speed));
              const timer = setTimeout(() => void get().advanceBackwardStage(stageIndex + 1), delay);
              set({ autoTimer: timer });
            }
          } catch {
            set({ isRunning: false, autoTimer: null });
          }
        },

        async fetchSaliency() {
          const response = await apiClient.post("/api/lab/saliency", {
            architecture: get().architecture,
            dataset: get().dataset,
            pixels: Array.from(get().inputPixels),
            trueLabel: get().trueLabel,
          });
          const data = response.data;
          return {
            inputGradient: new Float32Array(data.inputGradient ?? []),
            inputShape: Array.isArray(data.inputShape) ? data.inputShape : [1, 28, 28],
            absoluteMax: typeof data.absoluteMax === "number" ? data.absoluteMax : 0,
            topPixels: Array.isArray(data.topPixels) ? data.topPixels : [],
          } as SaliencyData;
        },

        async fetchWeightData(stageId) {
          const response = await apiClient.post("/api/lab/weights", {
            architecture: get().architecture,
            dataset: get().dataset,
            stageId,
            includeUntrained: get().comparisonMode === "untrained",
          });
          const data = response.data;
          return {
            stageId,
            weights: new Float32Array(data.weights ?? []),
            bias: data.bias ? new Float32Array(data.bias) : null,
            shape: Array.isArray(data.shape) ? data.shape : [],
            statistics: data.statistics,
            untrainedWeights: data.untrainedWeights ? new Float32Array(data.untrainedWeights) : undefined,
            untrainedStatistics: data.untrainedStatistics,
          } as WeightInspectionData;
        },
      };
    },
    { name: "lab-store-v2" },
  ),
);
