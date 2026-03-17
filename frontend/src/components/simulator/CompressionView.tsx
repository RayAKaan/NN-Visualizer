import React from "react";
import { useCompressionStore } from "../../store/compressionStore";
import { useSimulatorStore } from "../../store/simulatorStore";
import { NeuralPanel } from "@/design-system/components/NeuralPanel";
import { NeuralInput } from "@/design-system/components/NeuralInput";
import { NeuralSelect } from "@/design-system/components/NeuralSelect";
import { NeuralButton } from "@/design-system/components/NeuralButton";

export function CompressionView() {
  const graphId = useSimulatorStore((s) => s.graphId);
  const { loading, sparsity, targetDtype, prune, quant, sweep, setSparsity, setTargetDtype, runPrune, runQuant, runSweep } = useCompressionStore();

  return (
    <div className="sim-view">
      <NeuralPanel className="view-panel" variant="base">
        <div className="view-title">Compression Studio</div>
        <div className="view-actions">
          <div className="view-field">
            <label className="view-label">Sparsity</label>
            <NeuralInput type="number" step={0.05} min={0} max={0.95} value={sparsity} onChange={(e) => setSparsity(Number(e.target.value))} />
          </div>
          <div className="view-field">
            <label className="view-label">Quant</label>
            <NeuralSelect value={targetDtype} onChange={(e) => setTargetDtype(e.target.value as "int8" | "fp16")}>
              <option value="int8">INT8</option>
              <option value="fp16">FP16</option>
            </NeuralSelect>
          </div>
          <NeuralButton variant="primary" disabled={!graphId || loading} onClick={() => graphId && runPrune(graphId)}>
            Prune
          </NeuralButton>
          <NeuralButton variant="secondary" disabled={!graphId || loading} onClick={() => graphId && runQuant(graphId)}>
            Quantize
          </NeuralButton>
          <NeuralButton variant="secondary" disabled={!graphId || loading} onClick={() => graphId && runSweep(graphId, [0.0, 0.2, 0.4, 0.6, 0.8])}>
            Sweep
          </NeuralButton>
        </div>
        {!graphId && <div className="view-warning">Build a graph first.</div>}
      </NeuralPanel>

      <div className="view-grid">
        <NeuralPanel className="view-panel" variant="base">
          <div className="view-subtitle">Prune Stats</div>
          {prune ? (
            <div className="view-stack">
              <div className="view-text">Original params: {prune.original_params}</div>
              <div className="view-text">Remaining: {prune.remaining_params}</div>
              <div className="view-text">Sparsity achieved: {(prune.sparsity_achieved * 100).toFixed(1)}%</div>
            </div>
          ) : (
            <div className="view-empty">No prune results yet.</div>
          )}
        </NeuralPanel>
        <NeuralPanel className="view-panel" variant="base">
          <div className="view-subtitle">Quant Stats</div>
          {quant ? (
            <div className="view-stack">
              <div className="view-text">Original bytes: {quant.original_memory_bytes}</div>
              <div className="view-text">Quant bytes: {quant.quantized_memory_bytes}</div>
              <div className="view-text">Compression: {quant.compression_ratio.toFixed(2)}x</div>
            </div>
          ) : (
            <div className="view-empty">No quant results yet.</div>
          )}
        </NeuralPanel>
        <NeuralPanel className="view-panel" variant="base">
          <div className="view-subtitle">Sweep</div>
          {sweep ? (
            <div className="view-stack">
              {sweep.sweep_results.map((r) => (
                <div key={r.sparsity} className="view-text">s={r.sparsity.toFixed(2)} params={r.params}</div>
              ))}
            </div>
          ) : (
            <div className="view-empty">No sweep results yet.</div>
          )}
        </NeuralPanel>
      </div>
    </div>
  );
}
