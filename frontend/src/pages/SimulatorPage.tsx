import React, { useEffect, useState } from "react";
import { SimulatorLayout } from "../components/simulator/SimulatorLayout";
import { SimulatorToolbar } from "../components/simulator/SimulatorToolbar";
import { ArchitectureBuilder } from "../components/simulator/ArchitectureBuilder";
import { DatasetPanel } from "../components/simulator/DatasetPanel";
import { NetworkCanvas } from "../components/simulator/NetworkCanvas";
import { ForwardPassPanel } from "../components/simulator/ForwardPassPanel";
import { EquationPanel } from "../components/simulator/EquationPanel";
import { InspectorView } from "../components/simulator/InspectorView";
import { BackwardPassPanel } from "../components/simulator/BackwardPassPanel";
import { DebugPanel } from "../components/simulator/DebugPanel";
import { ReplayPanel } from "../components/simulator/ReplayPanel";
import { LiveMetricsView } from "../components/simulator/LiveMetricsView";
import { PlaygroundView } from "../components/simulator/PlaygroundView";
import { ActivationsView } from "../components/simulator/ActivationsView";
import { SequenceView } from "../components/simulator/SequenceView";
import { ComparisonView } from "../components/simulator/ComparisonView";
import { ProfilerView } from "../components/simulator/ProfilerView";
import { LandscapeView } from "../components/simulator/LandscapeView";
import { EmbeddingsView } from "../components/simulator/EmbeddingsView";
import { InterpretView } from "../components/simulator/InterpretView";
import { AdversarialView } from "../components/simulator/AdversarialView";
import { CompressionView } from "../components/simulator/CompressionView";
import { GenerativeView } from "../components/simulator/GenerativeView";
import { AugmentationView } from "../components/simulator/AugmentationView";
import { ExperimentsView } from "../components/simulator/ExperimentsView";
import { AssistantPanel } from "../components/simulator/AssistantPanel";
import { HyperparameterPanel } from "../components/simulator/HyperparameterPanel";
import { ImportExportPanel } from "../components/simulator/ImportExportPanel";
import { TrainingControlBar } from "../components/simulator/TrainingControlBar";
import { KeyboardShortcutsModal } from "../components/simulator/KeyboardShortcutsModal";
import { useComputationStore } from "../store/computationStore";
import { useSimulatorStore } from "../store/simulatorStore";

export default function SimulatorPage() {
  const activeView = useSimulatorStore((s) => s.activeView);
  const setActiveView = useSimulatorStore((s) => s.setActiveView);
  const selectedLayerIndex = useSimulatorStore((s) => s.selectedLayerIndex);
  const currentInput = useSimulatorStore((s) => s.currentInput);
  const fetchEquations = useComputationStore((s) => s.fetchEquations);
  const inspectWeights = useComputationStore((s) => s.inspectWeights);
  const inspectActivations = useComputationStore((s) => s.inspectActivations);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) return;
      if (event.key === "?" || (event.key === "/" && event.shiftKey)) {
        setShortcutsOpen((v) => !v);
        return;
      }
      if (event.key === "Escape") {
        setShortcutsOpen(false);
        return;
      }
      const key = event.key;
      const map: Record<string, typeof activeView> = {
        "1": "network",
        "2": "playground",
        "3": "metrics",
        "4": "inspector",
        "5": "activations",
        "6": "sequence",
        "7": "compare",
        "8": "profile",
        "9": "landscape",
        "0": "embeddings",
        "i": "interpret",
        "a": "adversarial",
        "c": "compress",
        "g": "generate",
        "u": "augment",
        "e": "experiments",
      };
      if (map[key]) setActiveView(map[key]);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [setActiveView]);

  useEffect(() => {
    void fetchEquations(selectedLayerIndex);
    void inspectWeights(selectedLayerIndex);
    if (currentInput) {
      void inspectActivations(selectedLayerIndex, currentInput);
    }
  }, [fetchEquations, inspectActivations, inspectWeights, selectedLayerIndex, currentInput]);

  return (
    <>
      <SimulatorLayout
        header={<SimulatorToolbar />}
        left={
          <>
            <ArchitectureBuilder />
            <DatasetPanel />
            <HyperparameterPanel />
            <ImportExportPanel />
          </>
        }
        center={
          activeView === "network" ? (
            <NetworkCanvas />
          ) : activeView === "metrics" ? (
            <LiveMetricsView />
          ) : activeView === "activations" ? (
            <ActivationsView />
          ) : activeView === "sequence" ? (
            <SequenceView />
          ) : activeView === "compare" ? (
            <ComparisonView />
          ) : activeView === "profile" ? (
            <ProfilerView />
          ) : activeView === "landscape" ? (
            <LandscapeView />
          ) : activeView === "embeddings" ? (
            <EmbeddingsView />
          ) : activeView === "interpret" ? (
            <InterpretView />
          ) : activeView === "adversarial" ? (
            <AdversarialView />
          ) : activeView === "compress" ? (
            <CompressionView />
          ) : activeView === "generate" ? (
            <GenerativeView />
          ) : activeView === "augment" ? (
            <AugmentationView />
          ) : activeView === "experiments" ? (
            <ExperimentsView />
          ) : activeView === "inspector" ? (
            <InspectorView />
          ) : activeView === "playground" ? (
            <PlaygroundView />
          ) : (
            <div className="p-4 text-sm text-slate-300">Playground view coming next.</div>
          )
        }
        right={
          <>
            <EquationPanel />
            <ForwardPassPanel />
            <BackwardPassPanel />
            <DebugPanel />
            <ReplayPanel />
          </>
        }
        footer={<TrainingControlBar />}
      />
      <AssistantPanel />
      <KeyboardShortcutsModal open={shortcutsOpen} onClose={() => setShortcutsOpen(false)} />
    </>
  );
}
