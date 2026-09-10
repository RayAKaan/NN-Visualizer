import { useEffect } from "react";
import { InputStage } from "../components/lab/InputStage";
import { LabHeader } from "../components/lab/LabHeader";
import { PassDirectionIndicator } from "../components/lab/PassDirectionIndicator";
import { PlaybackControls } from "../components/lab/PlaybackControls";
import { StagePipeline } from "../components/lab/StagePipeline";
import { BackwardStageCard } from "../components/lab/backward/BackwardStageCard";
import { LossComputationViz } from "../components/lab/backward/LossComputationViz";
import { TruthSelector } from "../components/lab/backward/TruthSelector";
import { ComparisonStageCard } from "../components/lab/comparison/ComparisonStageCard";
import { TrainedVsUntrainedToggle } from "../components/lab/comparison/TrainedVsUntrainedToggle";
import { WeightInspector } from "../components/lab/inspection/WeightInspector";
import { SaliencyOverlay } from "../components/lab/saliency/SaliencyOverlay";
import { ArchitectureComparison } from "../components/lab/comparison/ArchitectureComparison";
import { CounterfactualExplorer } from "../components/lab/counterfactual/CounterfactualExplorer";
import { DataFlowRibbon } from "../components/lab/flow/DataFlowRibbon";
import { NeuronBiographyPanel } from "../components/lab/neuron/NeuronBiographyPanel";
import { CostProfiler } from "../components/lab/profiler/CostProfiler";
import { useAutoScroll } from "../hooks/useAutoScroll";
import { useComparisonStore } from "../store/comparisonStore";
import { useCounterfactualStore } from "../store/counterfactualStore";
import { useFlowStore } from "../store/flowStore";
import { useLabStore } from "../store/labStore";
import { useNeuronStore } from "../store/neuronStore";
import { useProfilerStore } from "../store/profilerStore";
import type { Architecture, Dataset } from "../types/pipeline";
import { NeuralButton } from "@/design-system/components/NeuralButton";
import { WORKSPACE_EVENTS } from "../utils/workspaceEvents";

interface LabToolsBarProps {
  inputPixels: Float32Array;
  architecture: Architecture;
  dataset: Dataset;
  currentStageId: string | null;
  onStartComparison: () => void;
  onOpenCounterfactual: () => void;
}

function LabToolsBar({ inputPixels, architecture, dataset, currentStageId, onStartComparison, onOpenCounterfactual }: LabToolsBarProps) {
  const flowVisible = useFlowStore((s) => s.isRibbonVisible);
  const toggleFlow = useFlowStore((s) => s.toggleRibbon);
  const profilerVisible = useProfilerStore((s) => s.isProfilerVisible);
  const toggleProfiler = useProfilerStore((s) => s.toggleProfiler);
  const computeSaliency = useLabStore((s) => s.computeSaliency);
  const openNeuron = useNeuronStore((s) => s.openNeuron);
  const hasInput = inputPixels.some((value) => value > 0.05);

  return (
    <section className="lab-tools-bar" aria-labelledby="lab-tools-heading">
      <div className="lab-tools-intro">
        <span className="lab-section-kicker" id="lab-tools-heading">Explore</span>
        <p>Keep the core pass clear. Open a diagnostic only when it answers a question.</p>
      </div>
      <div className="lab-tools-actions">
        <NeuralButton size="sm" variant={flowVisible ? "primary" : "secondary"} onClick={toggleFlow}>
          {flowVisible ? "Hide flow" : "Show flow"}
        </NeuralButton>
        <NeuralButton size="sm" variant={profilerVisible ? "primary" : "secondary"} onClick={toggleProfiler}>
          {profilerVisible ? "Hide profiler" : "Cost profiler"}
        </NeuralButton>
        <NeuralButton size="sm" variant="secondary" disabled={!hasInput} onClick={() => void computeSaliency()}>
          Saliency
        </NeuralButton>
        <NeuralButton size="sm" variant="secondary" disabled={!hasInput} onClick={onOpenCounterfactual}>
          Counterfactual
        </NeuralButton>
        <NeuralButton size="sm" variant="secondary" disabled={!hasInput} onClick={onStartComparison}>
          Compare stages
        </NeuralButton>
        <NeuralButton
          size="sm"
          variant="secondary"
          disabled={!currentStageId || !hasInput}
          onClick={() => currentStageId && void openNeuron(currentStageId, 0, architecture, dataset, inputPixels)}
        >
          Inspect neuron
        </NeuralButton>
      </div>
    </section>
  );
}

export default function LabPage() {
  const currentStageIndex = useLabStore((s) => s.currentStageIndex);
  const ref = useAutoScroll(currentStageIndex);
  const setArchitecture = useLabStore((s) => s.setArchitecture);
  const setDataset = useLabStore((s) => s.setDataset);
  const resetPipeline = useLabStore((s) => s.resetPipeline);
  const passDirection = useLabStore((s) => s.passDirection);
  const lossInfo = useLabStore((s) => s.lossInfo);
  const stages = useLabStore((s) => s.stages);
  const backwardStatuses = useLabStore((s) => s.backwardStageStatuses);
  const backwardActivations = useLabStore((s) => s.backwardActivations);
  const comparisonMode = useLabStore((s) => s.comparisonMode);
  const activations = useLabStore((s) => s.activations);
  const untrainedActivations = useLabStore((s) => s.untrainedActivations);
  const saliencyData = useLabStore((s) => s.saliencyData);
  const weightInspection = useLabStore((s) => s.weightInspection);
  const clearWeightInspection = useLabStore((s) => s.clearWeightInspection);
  const architecture = useLabStore((s) => s.architecture);
  const dataset = useLabStore((s) => s.dataset);
  const inputPixels = useLabStore((s) => s.inputPixels);
  const flowVisible = useFlowStore((s) => s.isRibbonVisible);
  const profilerVisible = useProfilerStore((s) => s.isProfilerVisible);
  const openCounterfactual = useCounterfactualStore((s) => s.openExplorer);
  const closeCounterfactual = useCounterfactualStore((s) => s.closeExplorer);
  const startComparison = useComparisonStore((s) => s.startComparison);
  const stopComparison = useComparisonStore((s) => s.stopComparison);

  const backwardStages = stages.filter((s) => !["input", "output"].includes(s.type)).reverse();

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.repeat) return;
      const target = event.target as HTMLElement | null;
      if (target?.closest("input, textarea, select, [contenteditable=\"true\"]")) return;
      const state = useLabStore.getState();
      if (event.code === "Space") {
        event.preventDefault();
        if (state.currentStageIndex < 0) void state.startPipeline();
        else if (state.isRunning) state.pausePipeline();
        else state.resumePipeline();
      }
      if (event.key === "ArrowRight") void state.stepForward();
      if (event.key === "ArrowLeft") state.stepBackward();
      if (event.key === "Home") state.resetPipeline();
      if (event.key === "End") void state.skipToEnd();
      if (event.key === "1") setArchitecture("ANN");
      if (event.key === "2") setArchitecture("CNN");
      if (event.key === "3") setArchitecture("RNN");
      if (event.key.toLowerCase() === "m") setDataset("mnist");
      if (event.key.toLowerCase() === "c" && event.key !== "C") setDataset("catdog");
      if (event.key === "F" || event.key === "f") useFlowStore.getState().toggleRibbon();
      if (event.key === "P" || event.key === "p") useProfilerStore.getState().toggleProfiler();
      if (event.key === "W" || event.key === "w") openCounterfactual();
      if (event.key === "C") void startComparison(state.inputPixels, state.dataset);
      if (event.key === "N" || event.key === "n") {
        const stage = state.stages[state.currentStageIndex];
        if (stage) {
          void useNeuronStore
            .getState()
            .openNeuron(stage.id, 0, state.architecture, state.dataset, state.inputPixels);
        }
      }
      if (event.key === "+") state.setSpeed(Math.min(4, state.speed * 2));
      if (event.key === "-") state.setSpeed(Math.max(0.5, state.speed / 2));
      if (event.key === "Escape") {
        if (useNeuronStore.getState().isOpen) useNeuronStore.getState().closeNeuron();
        else if (useCounterfactualStore.getState().isOpen) closeCounterfactual();
        else if (useComparisonStore.getState().isComparisonActive) stopComparison();
        else if (useProfilerStore.getState().isProfilerVisible) useProfilerStore.getState().setProfilerVisible(false);
      }
    };

    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [resetPipeline, setArchitecture, setDataset]);

  useEffect(() => {
    const onResetCommand = () => resetPipeline();
    window.addEventListener(WORKSPACE_EVENTS.resetLab, onResetCommand);
    return () => window.removeEventListener(WORKSPACE_EVENTS.resetLab, onResetCommand);
  }, [resetPipeline]);

  return (
    <div className="flex h-full flex-col">
      <LabHeader />
      <div ref={ref} className="flex-1 overflow-y-auto pb-36">
        <div className="page-shell [--shell-max:72rem] py-4">
          {flowVisible ? <DataFlowRibbon /> : null}
          <div className="lab-orientation-row">
            <div>
              <span className="lab-section-kicker">Core experiment</span>
              <p className="lab-next-step">Input → current stage → next computation</p>
            </div>
            <div className="flex items-center gap-2">
              <PassDirectionIndicator />
              <TrainedVsUntrainedToggle />
            </div>
          </div>
          <LabToolsBar
            inputPixels={inputPixels}
            architecture={architecture}
            dataset={dataset}
            currentStageId={stages[currentStageIndex]?.id ?? null}
            onStartComparison={() => void startComparison(inputPixels, dataset)}
            onOpenCounterfactual={openCounterfactual}
          />
          <InputStage />
          <StagePipeline />
          <TruthSelector />
          {lossInfo ? <LossComputationViz lossInfo={lossInfo} /> : null}

          {passDirection === "backward" ? (
            <section className="mt-4 space-y-3">
              {backwardStages.map((stage, idx) => (
                <BackwardStageCard
                  key={`bwd-${stage.id}`}
                  stage={stage}
                  status={backwardStatuses[stage.id] ?? "locked"}
                  activation={backwardActivations[stage.id] ?? null}
                  stageNumber={idx + 1}
                />
              ))}
            </section>
          ) : null}

          {comparisonMode === "untrained" ? (
            <section className="mt-4 space-y-2">
              {stages.map((stage) => (
                activations[stage.id] ? (
                  <ComparisonStageCard
                    key={`cmp-${stage.id}`}
                    stage={stage}
                    trainedActivation={activations[stage.id]}
                    untrainedActivation={untrainedActivations[stage.id] ?? null}
                  />
                ) : null
              ))}
            </section>
          ) : null}

          {saliencyData ? <SaliencyOverlay saliencyData={saliencyData} /> : null}
          {profilerVisible ? <CostProfiler /> : null}
        </div>
      </div>
      <PlaybackControls />
      {weightInspection ? <WeightInspector data={weightInspection} onClose={clearWeightInspection} /> : null}
      <ArchitectureComparison />
      <CounterfactualExplorer />
      <NeuronBiographyPanel />
    </div>
  );
}
