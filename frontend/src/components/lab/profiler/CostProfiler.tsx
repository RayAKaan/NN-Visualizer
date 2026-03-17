import { useMemo } from "react";
import type { LayerProfile, NetworkProfile } from "../../../types/profiler";
import { useLabStore } from "../../../store/labStore";

function formatNumber(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}G`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return n.toFixed(0);
}

function formatBytes(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)} KB`;
  return `${n.toFixed(0)} B`;
}

function CostRibbon({ layer, isBottleneck }: { layer: LayerProfile; isBottleneck: boolean }) {
  return (
    <div className="flex items-center gap-2 py-1 text-[10px]">
      <span className="w-24 truncate" style={{ color: isBottleneck ? "var(--warning)" : "var(--text-3)" }}>
        {isBottleneck ? "! " : ""}
        {layer.stageId}
      </span>
      <div className="h-2 flex-1 overflow-hidden rounded" style={{ background: "var(--bg-void)" }}>
        <div className="h-full" style={{ width: `${layer.flopPercent}%`, background: "var(--fwd)" }} />
      </div>
      <span className="w-12 text-right font-mono" style={{ color: "var(--text-4)" }}>
        {layer.inferenceTimeMs.toFixed(1)}ms
      </span>
    </div>
  );
}

export function CostProfiler() {
  const architecture = useLabStore((s) => s.architecture);
  const stages = useLabStore((s) => s.stages);
  const activations = useLabStore((s) => s.activations);

  const profile = useMemo<NetworkProfile | null>(() => {
    const activeStages = stages.filter((s) => activations[s.id]);
    if (activeStages.length === 0) return null;

    const layers: LayerProfile[] = activeStages.map((stage) => {
      const act = activations[stage.id];
      const inputElems = Math.max(1, stage.inputShape.reduce((a, b) => a * b, 1));
      const outputElems = Math.max(1, stage.outputShape.reduce((a, b) => a * b, 1));
      const params = act.metadata.paramCount;
      const time = act.metadata.computeTimeMs;

      let flops = outputElems;
      if (stage.type === "dense") flops = 2 * inputElems * outputElems;
      if (stage.type === "conv2d") {
        const k = Number(stage.params?.kernelSize ?? 3);
        flops = 2 * outputElems * k * k;
      }
      if (stage.type === "softmax") flops = 3 * outputElems;

      const memoryBytes = outputElems * 4;
      const parameterBytes = params * 4;

      return {
        stageId: stage.id,
        flops,
        memoryBytes,
        parameterCount: params,
        parameterBytes,
        inferenceTimeMs: time,
        flopPercent: 0,
        memoryPercent: 0,
        paramPercent: 0,
        timePercent: 0,
        flopsPerParam: params > 0 ? flops / params : 0,
        memoryEfficiency: outputElems / Math.max(1, memoryBytes),
      };
    });

    const totalFlops = layers.reduce((a, b) => a + b.flops, 0);
    const totalMemoryBytes = layers.reduce((a, b) => a + b.memoryBytes, 0);
    const totalParams = layers.reduce((a, b) => a + b.parameterCount, 0);
    const totalInferenceMs = layers.reduce((a, b) => a + b.inferenceTimeMs, 0);

    for (const l of layers) {
      l.flopPercent = totalFlops ? (l.flops / totalFlops) * 100 : 0;
      l.memoryPercent = totalMemoryBytes ? (l.memoryBytes / totalMemoryBytes) * 100 : 0;
      l.paramPercent = totalParams ? (l.parameterCount / totalParams) * 100 : 0;
      l.timePercent = totalInferenceMs ? (l.inferenceTimeMs / totalInferenceMs) * 100 : 0;
    }

    const bottleneck = layers.reduce((a, b) => (b.flops > a.flops ? b : a), layers[0]);

    return {
      architecture,
      totalFlops,
      totalMemoryBytes,
      totalParams,
      totalInferenceMs,
      layers,
      bottleneckLayer: bottleneck.stageId,
      bottleneckType: "compute",
    };
  }, [activations, architecture, stages]);

  if (!profile) return null;

  return (
    <section className="mt-4 rounded-2xl p-3" style={{ background: "var(--bg-card)", border: "1px solid var(--glass-border)" }}>
      <h3 className="text-sm font-semibold" style={{ color: "var(--text-1)" }}>Computational Cost Profile</h3>
      <div className="mt-2 grid grid-cols-2 gap-2 text-xs md:grid-cols-4">
        <div className="rounded-lg p-2" style={{ background: "var(--bg-panel)" }}>FLOPs: {formatNumber(profile.totalFlops)}</div>
        <div className="rounded-lg p-2" style={{ background: "var(--bg-panel)" }}>Memory: {formatBytes(profile.totalMemoryBytes)}</div>
        <div className="rounded-lg p-2" style={{ background: "var(--bg-panel)" }}>Params: {formatNumber(profile.totalParams)}</div>
        <div className="rounded-lg p-2" style={{ background: "var(--bg-panel)" }}>Time: {profile.totalInferenceMs.toFixed(1)}ms</div>
      </div>

      <div className="mt-3 space-y-1">
        {profile.layers.map((layer) => (
          <CostRibbon key={layer.stageId} layer={layer} isBottleneck={layer.stageId === profile.bottleneckLayer} />
        ))}
      </div>
    </section>
  );
}
