import { useNeuronStore } from "../../../store/neuronStore";

export function NeuronBiographyPanel() {
  const isOpen = useNeuronStore((s) => s.isOpen);
  const neuron = useNeuronStore((s) => s.neuron);
  const isLoading = useNeuronStore((s) => s.isLoading);
  const error = useNeuronStore((s) => s.error);
  const close = useNeuronStore((s) => s.closeNeuron);

  if (!isOpen) return null;

  return (
    <aside
      className="fixed right-4 top-20 bottom-24 z-50 w-full overflow-y-auto rounded-2xl border shadow-2xl backdrop-blur-xl transition-all duration-300 md:w-[420px]"
      style={{ background: "var(--bg-card)", borderColor: "var(--glass-border)" }}
    >
      <div className="flex items-center justify-between border-b px-3 py-2" style={{ borderColor: "var(--glass-border)" }}>
        <h3 className="text-sm font-semibold" style={{ color: "var(--text-1)" }}>Neuron Biography</h3>
        <button type="button" className="text-xs" style={{ color: "var(--text-3)" }} onClick={close}>Close</button>
      </div>

      {isLoading ? <div className="p-3 text-xs" style={{ color: "var(--text-4)" }}>Loading�</div> : null}
      {error ? <div className="p-3 text-xs" style={{ color: "var(--error)" }}>{error}</div> : null}

      {neuron ? (
        <div className="space-y-2 p-3 text-xs">
          <div className="rounded-lg border p-2" style={{ borderColor: "var(--glass-border)", background: "var(--bg-panel)" }}>
            Stage: {neuron.stageId} � index: {neuron.neuronIndex} � type: {neuron.layerType}
          </div>
          <div className="rounded-lg border p-2" style={{ borderColor: "var(--glass-border)", background: "var(--bg-panel)" }}>
            Activation: {neuron.currentActivation.toFixed(5)} � importance: {(neuron.importanceScore * 100).toFixed(1)}%
          </div>
          <div className="rounded-lg border p-2" style={{ borderColor: "var(--glass-border)", background: "var(--bg-panel)" }}>
            Incoming: {neuron.incomingConnections} � Outgoing: {neuron.outgoingConnections} � Ablation impact: {(neuron.ablationImpact * 100).toFixed(2)}%
          </div>
        </div>
      ) : null}
    </aside>
  );
}
