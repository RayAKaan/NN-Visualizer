import { useMemo, useState } from "react";
import { useExecutionStore } from "../../../store/executionStore";
import type { TraceStage } from "../../../types/execution";
import { fmt, StatChips, TensorHeatmap, TensorStrip, TruthChip } from "./visual";

interface ViewProps {
  stage: TraceStage;
}

function Block({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-barley-line bg-barley-page/60 p-2.5">
      <div className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-ink-faint">{title}</div>
      {children}
    </div>
  );
}

export function InputView({ stage }: ViewProps) {
  const trace = useExecutionStore((s) => s.trace);
  const input = trace?.input.values ?? [];
  const shape = trace?.input.shape ?? [];
  const map = useMemo(() => {
    const [h = 28, w = 28] = shape.length === 1 ? [28, 28] : shape;
    const grid: number[][] = [];
    for (let r = 0; r < h; r++) grid.push(input.slice(r * w, r * w + w));
    return grid;
  }, [input, shape]);

  return (
    <div className="space-y-3">
      <Block title="Sample tensor">
        <div className="flex flex-wrap items-start gap-4">
          <TensorHeatmap map={map} cell={12} ariaLabel={`input ${map.length}×${map[0]?.length ?? 0}`} />
          <div>
            <div className="text-xs text-ink-mute">
              {shape.length === 1 ? "flattened signal, shown as rows" : "per-pixel intensity"}
            </div>
            <StatChips stats={stage.statistics} />
          </div>
        </div>
      </Block>
    </div>
  );
}

export function DenseView({ stage }: ViewProps) {
  const selection = useExecutionStore((s) => s.selection);
  const selectNeuron = useExecutionStore((s) => s.selectNeuron);
  const neuronDetail = useExecutionStore((s) => s.neuronDetail);
  const output = stage.output_data ?? [];
  const [draft, setDraft] = useState(selection.neuronIndex);

  return (
    <div className="space-y-3">
      <Block title="Output neurons (click to inspect)">
        <TensorStrip
          values={output}
          highlightIndex={selection.neuronIndex}
          onClick={(i) => { setDraft(i); selectNeuron(i); }}
          title={`${output.length} dense activations`}
          height={40}
        />
        <div className="mt-2 flex items-center gap-3">
          <label className="text-[11px] text-ink-faint">neuron</label>
          <input
            type="range"
            min={0}
            max={Math.max(0, (stage.outputShape[0] ?? 1) - 1)}
            value={Math.min(draft, Math.max(0, (stage.outputShape[0] ?? 1) - 1))}
            onChange={(e) => setDraft(Number(e.target.value))}
            onPointerUp={() => selectNeuron(draft)}
            onKeyUp={() => selectNeuron(draft)}
            className="flex-1 accent-[var(--accent-primary)]"
            aria-label="select neuron"
          />
          <span className="w-16 text-right font-mono text-xs text-ink">
            #{selection.neuronIndex}
          </span>
        </div>
      </Block>

      {neuronDetail && neuronDetail.layerId === stage.layerId ? (
        <Block title={`Neuron #${neuronDetail.neuronIndex} — what it computed`}>
          <div className="grid gap-2 sm:grid-cols-3">
            <Metric label="z (pre-activation)" value={fmt(neuronDetail.preActivation)} />
            <Metric label={`${neuronDetail.activation ?? "identity"}(z) = a`} value={fmt(neuronDetail.postActivation)} />
            <Metric label="bias" value={fmt(neuronDetail.bias)} />
          </div>
          <div className="mt-2 flex items-center gap-2">
            <TruthChip ok={neuronDetail.matchedOutput} label="verified against layer output" />
            <span className="text-[11px] text-ink-faint">
              contribution norm = {fmt(neuronDetail.weightRowStats.norm)}
            </span>
          </div>
          <div className="mt-2 space-y-1">
            {neuronDetail.topContributions.slice(0, 12).map((c) => (
              <div key={c.index} className="flex items-center gap-2 text-[11px]" title={`input ${c.index}`}>
                <span className="w-14 shrink-0 font-mono text-ink-faint">x[{c.index}]</span>
                <span className="w-16 shrink-0 font-mono text-ink-mute">{fmt(c.inputValue)}</span>
                <span className="w-5 shrink-0 text-center text-ink-faint">·</span>
                <span className="w-[72px] shrink-0 font-mono text-ink-mute">{fmt(c.weight)}</span>
                <span className="w-3 shrink-0 text-center text-ink-faint">=</span>
                <div
                  className="h-2.5 rounded min-w-[2px]"
                  style={{
                    width: `${Math.max(3, (Math.abs(c.contribution) / Math.max(neuronDetail.maxAbsContribution, 1e-9)) * 120)}px`,
                    background: c.contribution >= 0 ? "rgba(194,65,12,0.85)" : "rgba(0,114,178,0.85)",
                  }}
                />
                <span className="font-mono text-ink-soft">{fmt(c.contribution)}</span>
              </div>
            ))}
          </div>
        </Block>
      ) : null}
    </div>
  );
}

export function ConvView({ stage }: ViewProps) {
  const selection = useExecutionStore((s) => s.selection);
  const selectFilter = useExecutionStore((s) => s.selectFilter);
  const selectPosition = useExecutionStore((s) => s.selectPosition);
  const convDetail = useExecutionStore((s) => s.convDetail);
  const filters = stage.outputShape[2] ?? stage.feature_maps?.length ?? 0;
  const map = (stage.feature_maps?.[selection.filterIndex] ?? []) as number[][];

  return (
    <div className="space-y-3">
      <Block title="Feature map (click a cell → cell math)">
        <div className="mb-2 flex flex-wrap items-center gap-1">
          <span className="text-[11px] text-ink-faint">filter</span>
          {Array.from({ length: Math.min(filters, 96) }, (_, i) => (
            <button
              key={i}
              onClick={() => selectFilter(i)}
              className={`rounded-md border px-1.5 py-0.5 text-[11px] ${
                i === selection.filterIndex
                  ? "border-[var(--accent-primary)] bg-[color-mix(in_srgb,var(--accent-primary)_12%,white)] text-[var(--accent-primary-strong)] font-semibold ring-1 ring-[var(--accent-primary)]"
                  : "border-barley-line bg-barley-wash text-ink-mute hover:text-ink"
              }`}
              aria-pressed={i === selection.filterIndex}
            >
              {i}
            </button>
          ))}
        </div>
        {map.length ? (
          <TensorHeatmap
            map={map}
            cell={11}
            onCell={(_h, _w, value) => selectPosition(_h, _w)}
            ariaLabel={`filter ${selection.filterIndex} feature map`}
          />
        ) : (
          <div className="text-sm text-ink-faint">No feature map data for this filter.</div>
        )}

        {convDetail && convDetail.layerId === stage.layerId && convDetail.selectedChannel === selection.filterIndex ? (
          <div className="mt-2 grid gap-2 sm:grid-cols-3">
            <Metric label={`z at (${convDetail.position.h},${convDetail.position.w})`} value={fmt(convDetail.preActivation)} />
            <Metric label={`${convDetail.activation ?? "identity"}(z) = a`} value={fmt(convDetail.postActivation)} />
            <div className="flex items-center gap-2">
              <TruthChip ok={convDetail.matchedOutput} label="verified" detail="manual patch×kernel matches the layer output" />
            </div>
          </div>
        ) : null}
      </Block>
      <ConvMicroscope stage={stage} />
    </div>
  );
}

function ConvMicroscope({ stage }: ViewProps) {
  const convDetail = useExecutionStore((s) => s.convDetail);
  if (!convDetail || convDetail.layerId !== stage.layerId) return null;
  const patch = convDetail.patch;
  const kernel = convDetail.kernel;
  const inChannels = patch[0]?.[0]?.length ?? 1;
  return (
    <Block title={`Cell math — filter ${convDetail.filterIndex} at (${convDetail.position.h},${convDetail.position.w})`}>
      <div className="overflow-x-auto">
        <div className="flex items-start gap-4">
          <div>
            <div className="mb-1 text-center text-[10px] uppercase tracking-wide text-ink-faint">kernel ({kernel.length}×{kernel[0]?.length ?? 0})</div>
            <TensorHeatmap map={kernel} cell={26} ariaLabel="conv kernel weights" />
          </div>
          <div>
            <div className="mb-1 text-center text-[10px] uppercase tracking-wide text-ink-faint">input patch × kernel</div>
            <div className="grid gap-1.5">
              {patch.map((row, r) => (
                <div key={r} className="flex gap-1.5">
                  {row.map((cell, c) => {
                    const products = cell.map((pv, ch) => pv * (kernel[r]?.[c] ?? 0));
                    const sum = products.reduce((a, b) => a + b, 0);
                    const bg = sum > 0 ? "rgba(194,65,12,0.2)" : sum < 0 ? "rgba(0,114,178,0.2)" : "rgba(236,231,222,0.6)";
                    return (
                      <div
                        key={c}
                        className="grid place-items-center rounded-md border border-barley-line font-mono text-[10px] text-ink"
                        style={{ width: 52, height: 52, background: sum !== 0 ? bg : undefined }}
                        title={`(h${r},w${c}): Σ(x·k) = ${fmt(sum, 3)}`}
                      >
                        {fmt(sum, 2)}
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-ink-mute">
        <span>Σ(patch·kernel) = <b className="text-ink">{fmt(convDetail.preActivation - convDetail.bias, 3)}</b></span>
        <span>+ bias <b className="text-ink">{fmt(convDetail.bias)}</b></span>
        <span>= z <b className="text-ink">{fmt(convDetail.preActivation)}</b></span>
        <span>→ {convDetail.activation ?? "identity"}(z) = <b className="text-ink">{fmt(convDetail.postActivation)}</b></span>
        {inChannels > 1 ? <span className="text-ink-faint">({inChannels} input channels summed)</span> : null}
      </div>
    </Block>
  );
}

export function PoolView({ stage }: ViewProps) {
  const selection = useExecutionStore((s) => s.selection);
  const selectFilter = useExecutionStore((s) => s.selectFilter);
  const filters = stage.feature_maps?.length ?? 0;
  const map = (stage.feature_maps?.[selection.filterIndex] ?? []) as number[][];
  return (
    <div className="space-y-3">
      <Block title={`${stage.operation === "max_pool" ? "Max" : "Avg"} pooling — downsample preserves the strongest signal`}>
        <div className="mb-2 flex flex-wrap gap-1">
          <span className="text-[11px] text-ink-faint">channel</span>
          {Array.from({ length: filters }, (_, i) => (
            <button
              key={i}
              onClick={() => selectFilter(i)}
              className={`rounded-md border px-1.5 py-0.5 text-[11px] ${
                i === selection.filterIndex ? "border-[var(--accent-tertiary)] bg-[color-mix(in_srgb,var(--accent-tertiary)_12%,white)] text-[#065f52] font-semibold" : "border-barley-line bg-barley-wash text-ink-mute"
              }`}
              aria-pressed={i === selection.filterIndex}
            >
              {i}
            </button>
          ))}
        </div>
        {map.length ? <TensorHeatmap map={map} cell={14} ariaLabel={`channel ${selection.filterIndex} pooled map`} /> : null}
      </Block>
    </div>
  );
}

export function ActivationView({ stage }: ViewProps) {
  const output = stage.output_data ?? [];
  const dead = stage.statistics.zero;
  return (
    <div className="space-y-3">
      <Block title={`${stage.activation ?? "activation"} applied pointwise`}>
        <TensorStrip values={output} height={34} title={`${stage.activation ?? ""} activation values`} />
        <div className="mt-2 flex flex-wrap gap-x-4 text-[11px] text-ink-mute">
          {stage.activation === "relu" ? (
            <span>dead (output ≤ 0) fraction: <b className="text-ink">{fmt(dead * 100, 1)}%</b></span>
          ) : (
            <span>pass-through of the previous layer — no units changed.</span>
          )}
        </div>
      </Block>
    </div>
  );
}

export function SoftmaxView({ stage }: ViewProps) {
  const probs = stage.output_data ?? [];
  const pred = useExecutionStore((s) => s.trace?.prediction);
  const maxAbs = Math.max(...probs, 1e-6);
  return (
    <div className="space-y-3">
      <Block title="Class probabilities">
        <div className="space-y-1">
          {probs.map((p, i) => {
            const isPred = pred && Number(pred.label) === i;
            return (
              <div key={i} className="flex items-center gap-2 text-[12px]">
                <span className={`w-7 shrink-0 font-mono ${isPred ? "font-bold text-[var(--accent-primary-strong)]" : "text-ink-mute"}`}>{i}</span>
                <div className="h-3.5 flex-1 rounded bg-barley-wash">
                  <div
                    className={`h-full rounded ${isPred ? "bg-[var(--accent-primary)]" : "bg-[rgba(194,65,12,0.55)]"}`}
                    style={{ width: `${(p / maxAbs) * 100}%` }}
                  />
                </div>
                <span className="w-16 shrink-0 text-right font-mono text-ink-soft">{(p * 100).toFixed(1)}%</span>
              </div>
            );
          })}
        </div>
      </Block>
    </div>
  );
}

export function FlattenView({ stage }: ViewProps) {
  const output = stage.output_data ?? [];
  return (
    <div className="space-y-3">
      <Block title={`Flatten: ${stage.inputShape.join("×")} → ${stage.outputShape.join("×")} = ${output.length} values`}>
        <TensorStrip values={output} height={22} title="flattened values" />
        <div className="mt-2 text-[11px] text-ink-mute">
          pure reshape — the same numbers, relinearized. Norm&nbsp;=
          <b className="text-ink">&nbsp;{fmt(Math.sqrt(output.reduce((a, b) => a + b * b, 0)))}</b>
        </div>
      </Block>
    </div>
  );
}

export function DropoutView({ stage }: ViewProps) {
  const output = stage.output_data ?? [];
  return (
    <div className="space-y-3">
      <Block title="Dropout at inference">
        <TensorStrip values={output} height={22} title="dropout pass-through" />
        <div className="mt-2 flex items-center gap-2 text-[11px] text-ink-mute">
          <TruthChip ok label="inference: identity (y = x)" detail="Dropout mutes units only during training; at inference it is a pass-through." />
          <span>input = output</span>
        </div>
      </Block>
    </div>
  );
}

const GATE_COLORS = {
  input: "rgba(0,178,146,0.85)",
  forget: "rgba(0,114,178,0.85)",
  candidate: "rgba(166,77,133,0.85)",
  output: "rgba(194,65,12,0.85)",
} as const;

export function LstmView({ stage }: ViewProps) {
  const selection = useExecutionStore((s) => s.selection);
  const selectTimestep = useExecutionStore((s) => s.selectTimestep);
  const detail = useExecutionStore((s) => s.lstmDetail);
  const [unit, setUnit] = useState(0);

  if (!detail) {
    return <div className="text-sm text-ink-faint">Loading real gate dynamics…</div>;
  }
  const summary = detail.summary;
  const T = detail.timesteps;
  const detailed = detail.detailed;

  return (
    <div className="space-y-3">
      <Block title="Gate activity across timesteps (real cell math)">
        <div className="mb-1 flex flex-wrap items-center gap-2">
          <label className="text-[11px] text-ink-faint">t</label>
          <input
            type="range"
            min={0}
            max={T - 1}
            value={selection.timestep}
            onChange={(e) => selectTimestep(Number(e.target.value))}
            className="flex-1 accent-[var(--accent-quaternary)]"
            aria-label="timestep"
          />
          <span className="w-14 text-right font-mono text-xs">{selection.timestep} / {T - 1}</span>
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          <Sparkline label="forget μ" data={summary.forgetMean} color={GATE_COLORS.forget} />
          <Sparkline label="input μ" data={summary.inputMean} color={GATE_COLORS.input} />
          <Sparkline label="output μ" data={summary.outputMean} color={GATE_COLORS.output} />
          <Sparkline label="‖hidden‖" data={summary.hiddenNorm} color="#1c1917" />
        </div>
        <div className="mt-2 flex items-center gap-2 text-[11px]">
          <TruthChip
            ok={detail.verified}
            label={detail.verified ? "manual cell matches Keras" : "manual deviation detected"}
            detail={`max |Δ final hidden state| = ${fmt(detail.maxDeviation)}`}
          />
          <span className="text-ink-faint">max deviation {fmt(detail.maxDeviation)}</span>
        </div>
      </Block>

      {detailed ? (
        <Block title={`Timestep ${detailed.timestep} — per-unit gates`}>
          <div className="mb-1 flex items-center gap-3">
            <span className="text-[11px] text-ink-faint">gate</span>
            {(["input", "forget", "candidate", "output"] as const).map((g) => (
              <span key={g} className="flex items-center gap-1 text-[11px] text-ink-mute">
                <span className="h-2 w-2 rounded-full" style={{ background: GATE_COLORS[g] }} />
                {g}
              </span>
            ))}
          </div>
          <TensorStrip values={detailed.gates.input} signedPositiveColor={GATE_COLORS.input} signedNegativeColor={GATE_COLORS.input} highlightIndex={unit} height={14} maxBarCount={256} />
          <TensorStrip values={detailed.gates.forget} signedPositiveColor={GATE_COLORS.forget} signedNegativeColor={GATE_COLORS.forget} highlightIndex={unit} height={14} maxBarCount={256} />
          <TensorStrip values={detailed.gates.candidate} signedPositiveColor={GATE_COLORS.candidate} signedNegativeColor={GATE_COLORS.candidate} highlightIndex={unit} height={14} maxBarCount={256} />
          <TensorStrip values={detailed.gates.output} signedPositiveColor={GATE_COLORS.output} signedNegativeColor={GATE_COLORS.output} highlightIndex={unit} height={14} maxBarCount={256} />
          <div className="mt-2 flex items-center gap-3">
            <label className="text-[11px] text-ink-faint">unit</label>
            <input
              type="range"
              min={0}
              max={Math.max(0, stage.outputShape[0] - 1)}
              value={unit}
              onChange={(e) => setUnit(Number(e.target.value))}
              className="flex-1 accent-[var(--accent-quaternary)]"
              aria-label="unit"
            />
            <span className="w-14 text-right font-mono text-xs">u{unit}</span>
          </div>
          <div className="mt-1 grid grid-cols-2 gap-x-4 gap-y-0.5 sm:grid-cols-4">
            {(["input", "forget", "candidate", "output"] as const).map((g) => (
              <div key={g} className="text-[11px] text-ink-mute">
                {g}: <b className="font-mono text-ink">{fmt(detailed.gates[g][unit], 4)}</b>
              </div>
            ))}
          </div>
        </Block>
      ) : null}
    </div>
  );
}

function Sparkline({ label, data, color, height = 34 }: { label: string; data: number[]; color: string; height?: number }) {
  const w = Math.max(data.length, 2);
  const max = Math.max(...data, 1e-6);
  const min = Math.min(...data, 0);
  const range = Math.max(max - min, 1e-6);
  const pts = data.map((v, i) => `${(i / (w - 1)) * 100},${height - 2 - ((v - min) / range) * (height - 6)}`).join(" ");
  return (
    <div className="rounded-lg border border-barley-line bg-ink/45 p-1.5">
      <div className="flex items-center justify-between text-[10px] uppercase tracking-wide text-ink-faint">
        <span>{label}</span>
        <span className="font-mono normal-case text-ink-mute">{fmt(data[data.length - 1] ?? 0, 3)}</span>
      </div>
      <svg viewBox={`0 0 100 ${height}`} preserveAspectRatio="none" className="h-8 w-full">
        {data.length > 1 ? <polyline points={pts} fill="none" stroke={color} strokeWidth={1.5} vectorEffect="non-scaling-stroke" /> : null}
      </svg>
    </div>
  );
}

export function StageDispatcher({ stage }: ViewProps) {
  const op = stage.operation;
  if (op === "input") return <InputView stage={stage} />;
  if (op === "dense") return stage.activation === "softmax" ? <SoftmaxView stage={stage} /> : <DenseView stage={stage} />;
  if (op === "conv2d") return <ConvView stage={stage} />;
  if (op === "max_pool" || op === "avg_pool") return <PoolView stage={stage} />;
  if (op === "flatten") return <FlattenView stage={stage} />;
  if (op === "activation") return <ActivationView stage={stage} />;
  if (op === "dropout") return <DropoutView stage={stage} />;
  if (op === "lstm" || op === "lstm_bidirectional") return <LstmView stage={stage} />;
  if (op === "softmax") return <SoftmaxView stage={stage} />;
  return (
    <div className="text-sm text-ink-faint">
      Operation <code className="rounded bg-barley-wash px-1">{op}</code> — values shown in the inspector.
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-barley-line bg-barley-wash px-2 py-1.5">
      <div className="text-[10px] uppercase tracking-wide text-ink-faint">{label}</div>
      <div className="font-mono text-base text-ink">{value}</div>
    </div>
  );
}