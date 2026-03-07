import React from "react";
import { BlockMath, InlineMath } from "react-katex";
import "katex/dist/katex.min.css";
import { useNeurofluxStore } from "../../store/useNeurofluxStore";
import { IntrospectionMode, NeuronState } from "./types";

interface Props {
  neuron: NeuronState | null;
  mode: IntrospectionMode;
}

function clamp01(v: number) {
  return Math.max(0, Math.min(1, v));
}

export default function NeuronInspector({ neuron, mode }: Props) {
  const currentArchitecture = useNeurofluxStore((s) => s.currentArchitecture);
  const playbackState = useNeurofluxStore((s) => s.playbackState);
  const perturbWeight = useNeurofluxStore((s) => s.perturbWeight);

  if (!neuron) {
    return (
      <div className="h-full rounded-xl border border-slate-700 bg-slate-900 p-4">
        <h3 className="text-lg font-semibold text-cyan-300">Neuron Inspector</h3>
        <p className="mt-3 text-sm text-slate-400">Select a neuron in the graph to inspect forward-pass and gradient equations.</p>
      </div>
    );
  }

  const incoming = neuron.incomingEdges.slice(0, 6);
  const expansion = incoming.slice(0, 3).map((edge, idx) => `${edge.weight.toFixed(2)}\\cdot x_${idx + 1}`).join(" + ");

  const sampleA = incoming.length > 0 ? incoming[0].contribution / (incoming[0].weight === 0 ? 1 : incoming[0].weight) : 0.72;
  const sampleDelta = neuron.gradient === 0 ? 0.0142 : neuron.gradient;
  const sampleGrad = sampleA * sampleDelta;
  const eta = 0.001;
  const firstWeight = incoming[0]?.weight ?? 0.45;
  const updatedWeight = firstWeight - eta * sampleGrad;
  const kernelWeights = neuron.layerType === "conv" ? neuron.incomingEdges.slice(0, 9).map((e) => e.weight) : [];
  const rw = neuron.incomingEdges[0]?.weight ?? 0.31;
  const ru = neuron.incomingEdges[1]?.weight ?? -0.22;
  const canPerturb = currentArchitecture === "ann" && playbackState === "paused" && neuron.layerType !== "input";
  const isOutputNeuron = /^O_\d+$/.test(neuron.id);
  const isLstmNode = neuron.layerType === "recurrent" && neuron.id.startsWith("LSTM_");
  const outputProb = clamp01(neuron.activation);
  const ceLoss = -Math.log(Math.max(1e-9, outputProb));

  return (
    <div className="h-full overflow-auto rounded-xl border border-slate-700 bg-slate-900 p-4 space-y-4">
      <div className="border-b border-slate-700 pb-3">
        <h3 className="text-lg font-semibold text-cyan-300">Neuron Inspector</h3>
        <p className="text-xs text-slate-400 mt-1">
          Node <span className="font-mono text-slate-200">{neuron.id}</span> | mode: <span className="uppercase text-emerald-300">{mode}</span>
        </p>
      </div>

      {isLstmNode ? (
        <section className="rounded-lg border border-amber-700/40 bg-amber-950/20 p-3">
          <h4 className="text-sm font-semibold text-amber-200">LSTM Temporal State</h4>
          <div className="mt-2 text-slate-300 text-sm">
            <BlockMath math={"f_t=\\sigma(W_f[h_{t-1},x_t]+b_f)"} />
            <BlockMath math={"i_t=\\sigma(W_i[h_{t-1},x_t]+b_i)"} />
            <BlockMath math={"\\tilde{C}_t=\\tanh(W_C[h_{t-1},x_t]+b_C)"} />
            <BlockMath math={"C_t=f_t \\odot C_{t-1}+i_t \\odot \\tilde{C}_t"} />
            <BlockMath math={"h_t=o_t \\odot \\tanh(C_t)"} />
            <div className="mt-2 text-sm text-slate-300">Current hidden activation: <InlineMath math={`h_t = ${neuron.activation.toFixed(4)}`} /></div>
          </div>
        </section>
      ) : neuron.layerType === "conv" ? (
        <section className="rounded-lg border border-sky-700/40 bg-sky-950/20 p-3">
          <h4 className="text-sm font-semibold text-sky-200">Convolutional Forward Pass</h4>
          <div className="mt-2 text-slate-300 text-sm">
            <BlockMath math={"z(i,j)=\\sum_{u,v}(K_{u,v}\\cdot X_{i+u,j+v})+b"} />
            <BlockMath math={"a(i,j)=\\mathrm{ReLU}(z(i,j))"} />
            <div className="mt-2 text-xs text-slate-400">Kernel weights (3x3)</div>
            <BlockMath
              math={`K=\\begin{bmatrix}${(kernelWeights[0] ?? 0.18).toFixed(2)} & ${(kernelWeights[1] ?? -0.04).toFixed(2)} & ${(kernelWeights[2] ?? 0.11).toFixed(2)}\\\\${(kernelWeights[3] ?? 0.09).toFixed(2)} & ${(kernelWeights[4] ?? 0.22).toFixed(2)} & ${(kernelWeights[5] ?? -0.07).toFixed(2)}\\\\${(kernelWeights[6] ?? -0.03).toFixed(2)} & ${(kernelWeights[7] ?? 0.14).toFixed(2)} & ${(kernelWeights[8] ?? 0.19).toFixed(2)}\\end{bmatrix}`}
            />
            <div className="mt-2 text-sm text-slate-300">Current activation: <InlineMath math={`a(i,j) = ${neuron.activation.toFixed(4)}`} /></div>
          </div>
        </section>
      ) : neuron.layerType === "recurrent" ? (
        <section className="rounded-lg border border-indigo-700/40 bg-indigo-950/20 p-3">
          <h4 className="text-sm font-semibold text-indigo-200">RNN Temporal State</h4>
          <div className="mt-2 text-slate-300 text-sm">
            <BlockMath math={"h_t=\\tanh(W_xx_t+W_hh_{t-1}+b_h)"} />
            <BlockMath math={"o_t=\\sigma(W_yh_t)"} />
            <div className="mt-2 text-xs text-slate-400">Expanded example</div>
            <BlockMath math={`h_t=\\tanh(${rw.toFixed(2)}x_t + ${ru.toFixed(2)}h_{t-1} + (${neuron.bias.toFixed(2)}))`} />
            <div className="mt-2 text-sm text-slate-300">Current hidden activation: <InlineMath math={`h_t = ${neuron.activation.toFixed(4)}`} /></div>
          </div>
        </section>
      ) : (
        <section className="rounded-lg border border-slate-700 bg-slate-950/40 p-3">
          <h4 className="text-sm font-semibold text-slate-200">Forward Pass</h4>
          <div className="mt-2 text-slate-300 text-sm">
            <BlockMath math={"z = \\sum_{i=1}^{n} w_i a_i + b"} />
            <BlockMath math={"a = \\mathrm{ReLU}(z) = \\max(0, z)"} />
            <div className="mt-2 text-xs text-slate-400">Expanded example</div>
            <BlockMath math={`z = ${expansion || "0.45\\cdot x_1 + 0.31\\cdot x_2 + -0.21\\cdot x_3"} + (${neuron.bias.toFixed(2)})`} />
            <div className="mt-2 text-sm text-slate-300">Current activation: <InlineMath math={`a = ${neuron.activation.toFixed(4)}`} /></div>
          </div>
        </section>
      )}

      {isOutputNeuron && (
        <section className="rounded-lg border border-emerald-700/40 bg-emerald-950/20 p-3">
          <h4 className="text-sm font-semibold text-emerald-200">Output Decision Math</h4>
          <div className="mt-2 text-slate-300 text-sm">
            <BlockMath math={"P(y=k)=\\frac{e^{z_k}}{\\sum_{j} e^{z_j}}"} />
            <BlockMath math={"L = -\\log(P(y=k))"} />
            <BlockMath math={`P(y=${neuron.id.split("_")[1]}) = ${outputProb.toFixed(5)}`} />
            <BlockMath math={`L = -\\log(${outputProb.toFixed(5)}) = ${ceLoss.toFixed(5)}`} />
          </div>
        </section>
      )}

      {neuron.incomingEdges.length > 0 && (
        <section className="rounded-lg border border-cyan-700/40 bg-cyan-950/20 p-3">
          <div className="flex items-center justify-between gap-3">
            <h4 className="text-sm font-semibold text-cyan-200">God Mode Weight Perturbation</h4>
            <span className="text-[11px] text-slate-400">{canPerturb ? "Paused ANN snapshot: editable" : "Pause ANN replay to edit weights"}</span>
          </div>
          <div className="mt-3 space-y-2">
            {neuron.incomingEdges.map((edge, idx) => (
              <div key={edge.id} className="grid grid-cols-[70px_1fr_56px] items-center gap-2 text-xs">
                <span className="font-mono text-slate-300">w{idx + 1}</span>
                <input
                  type="range"
                  min={-2}
                  max={2}
                  step={0.01}
                  value={edge.weight}
                  disabled={!canPerturb}
                  onChange={(e) => perturbWeight(edge.id, Number(e.target.value))}
                  className="accent-cyan-400 disabled:opacity-40"
                />
                <span className="font-mono text-cyan-300 text-right">{edge.weight.toFixed(2)}</span>
              </div>
            ))}
          </div>
        </section>
      )}

      {mode === "training" && (
        <>
          {neuron.layerType === "recurrent" && (
            <section className="rounded-lg border border-orange-700/40 bg-orange-950/20 p-3">
              <h4 className="text-sm font-semibold text-orange-200">BPTT Gradient Flow</h4>
              <div className="mt-2 text-slate-300 text-sm">
                <BlockMath math={"\\frac{\\partial L}{\\partial h_t}=\\frac{\\partial L}{\\partial h_T}\\prod_{k=t+1}^{T}\\frac{\\partial h_k}{\\partial h_{k-1}}"} />
                <BlockMath math={"\\frac{\\partial L}{\\partial W_h}=\\sum_t \\frac{\\partial L}{\\partial h_t}\\frac{\\partial h_t}{\\partial W_h}"} />
              </div>
            </section>
          )}
          <section className="rounded-lg border border-fuchsia-700/40 bg-fuchsia-950/20 p-3">
            <h4 className="text-sm font-semibold text-fuchsia-200">Backpropagation & Update</h4>
            <div className="mt-2 text-slate-300 text-sm">
              <BlockMath math={"\\frac{\\partial L}{\\partial w_{ij}} = a_i \\cdot \\delta_j"} />
              <BlockMath math={"w_{new} = w_{old} - \\eta\\frac{\\partial L}{\\partial w_{ij}}"} />
              <div className="mt-2 text-xs text-slate-400">Dummy calculated values</div>
              <BlockMath math={`\\frac{\\partial L}{\\partial w_{ij}} = ${sampleA.toFixed(4)} \\cdot ${sampleDelta.toFixed(6)} = ${sampleGrad.toFixed(6)}`} />
              <BlockMath math={`w_{new} = ${firstWeight.toFixed(4)} - ${eta.toFixed(4)}\\cdot ${sampleGrad.toFixed(6)} = ${updatedWeight.toFixed(6)}`} />
            </div>
          </section>
        </>
      )}
    </div>
  );
}
