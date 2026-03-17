import React from "react";
import { useSimulatorStore, SimulatorView } from "../../store/simulatorStore";
import { NeuralPanel } from "@/design-system/components/NeuralPanel";
import { NeuralTabs } from "@/design-system/components/NeuralTabs";

const tabs: Array<{ id: SimulatorView; label: string }> = [
  { id: "network", label: "Network" },
  { id: "playground", label: "Playground" },
  { id: "inspector", label: "Inspector" },
  { id: "metrics", label: "Metrics" },
  { id: "activations", label: "Activations" },
  { id: "sequence", label: "Sequence" },
  { id: "compare", label: "Compare" },
  { id: "profile", label: "Profile" },
  { id: "landscape", label: "Landscape" },
  { id: "embeddings", label: "Embeddings" },
  { id: "interpret", label: "Interpret" },
  { id: "adversarial", label: "Adversarial" },
  { id: "compress", label: "Compress" },
  { id: "generate", label: "Generate" },
  { id: "augment", label: "Augment" },
  { id: "experiments", label: "Experiments" },
];

export function SimulatorToolbar() {
  const activeView = useSimulatorStore((s) => s.activeView);
  const setActiveView = useSimulatorStore((s) => s.setActiveView);

  return (
    <NeuralPanel className="sim-toolbar" variant="base">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="sim-title">Neural Network Simulation Lab</div>
          <div className="sim-subtitle">Cinematic research workspace</div>
        </div>
        <div className="sim-tabs-wrap neural-scroll-area">
          <NeuralTabs
            tabs={tabs}
            value={activeView}
            onChange={(id) => setActiveView(id as SimulatorView)}
            className="sim-tabs"
          />
        </div>
      </div>
    </NeuralPanel>
  );
}
