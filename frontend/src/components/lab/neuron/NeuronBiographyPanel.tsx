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
      className="fixed right-4 top-20 bottom-24 z-50 w-full overflow-y-auto rounded-2xl border border-ink/10 bg-white shadow-pop backdrop-blur-xl transition-all duration-300 md:w-[420px]"
    >
      <div className="flex items-center justify-between border-b border-barley-line px-3 py-2">
        <h3 className="text-sm font-semibold text-ink">Neuron Biography</h3>
        <button type="button" className="text-xs text-ink-mute hover:text-ember-700" onClick={close}>Close</button>
      </div>

      {isLoading ? <div className="p-3 text-xs text-ink-faint">Loading{"\u2026"}</div> : null}
      {error ? <div className="p-3 text-xs text-status-danger">{error}</div> : null}

      {neuron ? (
        <div className="space-y-2 p-3 text-xs">
          <div className="rounded-lg border border-barley-linestrong bg-barley-page p-2">
            Stage: {neuron.stageId} {"\u00b7"} index: {neuron.neuronIndex} {"\u00b7"} type: {neuron.layerType}
          </div>
          <div className="rounded-lg border border-barley-linestrong bg-barley-page p-2">
            Activation: {neuron.currentActivation.toFixed(5)} {"\u00b7"} importance: {(neuron.importanceScore * 100).toFixed(1)}%
          </div>
          <div className="rounded-lg border border-barley-linestrong bg-barley-page p-2">
            Incoming: {neuron.incomingConnections} {"\u00b7"} Outgoing: {neuron.outgoingConnections} {"\u00b7"} Ablation impact: {(neuron.ablationImpact * 100).toFixed(2)}%
          </div>
        </div>
      ) : null}
    </aside>
  );
}
