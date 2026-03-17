import React, { useEffect, useState } from "react";
import { useExperimentStore } from "../../store/experimentStore";
import { useArchitectureStore } from "../../store/architectureStore";
import { NeuralPanel } from "@/design-system/components/NeuralPanel";
import { NeuralInput } from "@/design-system/components/NeuralInput";
import { NeuralButton } from "@/design-system/components/NeuralButton";

export function ExperimentsView() {
  const [name, setName] = useState("New Experiment");
  const layers = useArchitectureStore((s) => s.layers);
  const { experiments, loading, fetchAll, create, remove } = useExperimentStore();

  useEffect(() => {
    void fetchAll();
  }, [fetchAll]);

  return (
    <div className="sim-view">
      <NeuralPanel className="view-panel" variant="base">
        <div className="view-title">Experiments</div>
        <div className="view-actions">
          <NeuralInput value={name} onChange={(e) => setName(e.target.value)} />
          <NeuralButton variant="primary" disabled={loading} onClick={() => create({ name, config: { layers } })}>
            Save
          </NeuralButton>
        </div>
      </NeuralPanel>

      <NeuralPanel className="view-panel" variant="base">
        <div className="view-subtitle">Saved Experiments</div>
        <div className="view-stack">
          {experiments.map((exp) => (
            <div key={exp.id} className="view-row">
              <div>
                <div className="view-text strong">{exp.name}</div>
                <div className="view-meta">{exp.id}</div>
              </div>
              <NeuralButton variant="secondary" onClick={() => remove(exp.id)}>
                Delete
              </NeuralButton>
            </div>
          ))}
          {experiments.length === 0 && <div className="view-empty">No experiments yet.</div>}
        </div>
      </NeuralPanel>
    </div>
  );
}
