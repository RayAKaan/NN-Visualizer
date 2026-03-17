import type { BackwardStageActivation, StageDefinition } from "../../../types/pipeline";

interface Props {
  activation: BackwardStageActivation;
  stage: StageDefinition;
}

function chip(label: string, value: string) {
  return (
    <div className="rounded-lg border px-2 py-1" style={{ borderColor: "var(--glass-border)", background: "var(--bg-panel)" }}>
      <div className="text-[10px] uppercase" style={{ color: "var(--text-4)" }}>{label}</div>
      <div className="text-xs font-mono" style={{ color: "var(--text-2)" }}>{value}</div>
    </div>
  );
}

export function GradientFlowViz({ activation, stage }: Props) {
  const inSample = Array.from(activation.inputGradient.slice(0, 96));
  const outSample = Array.from(activation.outputGradient.slice(0, 96));
  const max = Math.max(...inSample.map((v) => Math.abs(v)), ...outSample.map((v) => Math.abs(v)), 0.0001);

  return (
    <div className="rounded-xl border p-3" style={{ borderColor: "var(--glass-border)", background: "rgba(0,0,0,0.15)" }}>
      <div className="mb-2 text-xs font-semibold uppercase" style={{ color: "var(--bwd)" }}>Gradient Flow</div>
      <div className="mb-3 flex flex-wrap gap-2">
        {chip("Flow", `${activation.stats.gradientFlowPercent.toFixed(1)}%`)}
        {chip("Norm", activation.stats.gradientNorm.toFixed(4))}
        {chip("Mean", activation.stats.inputGradMean.toFixed(6))}
        {chip("Max", activation.stats.inputGradMax.toFixed(4))}
        {activation.stats.deadNeuronPercent !== undefined ? chip("Dead", `${activation.stats.deadNeuronPercent.toFixed(1)}%`) : null}
      </div>

      <div className="text-[11px]" style={{ color: "var(--text-4)" }}>Output gradient</div>
      <div className="mt-1 flex h-6 gap-px overflow-hidden rounded">
        {outSample.map((v, i) => (
          <div key={`o-${i}`} className="flex-1" style={{ background: `rgba(251,146,60,${Math.abs(v) / max})` }} />
        ))}
      </div>
      <div className="my-1 text-center text-xs" style={{ color: "var(--bwd)" }}>× local derivative</div>
      <div className="text-[11px]" style={{ color: "var(--text-4)" }}>Input gradient</div>
      <div className="mt-1 flex h-6 gap-px overflow-hidden rounded">
        {inSample.map((v, i) => (
          <div key={`i-${i}`} className="flex-1" style={{ background: `rgba(251,146,60,${Math.abs(v) / max})` }} />
        ))}
      </div>

      {stage.type === "activation_relu" && activation.stats.deadNeuronPercent !== undefined && (
        <div className="mt-2 rounded-md px-2 py-1 text-xs" style={{ background: "var(--status-warning-bg)", color: "var(--status-warning)" }}>
          ReLU blocked {activation.stats.deadNeuronPercent.toFixed(1)}% of neuron gradients.
        </div>
      )}
    </div>
  );
}
