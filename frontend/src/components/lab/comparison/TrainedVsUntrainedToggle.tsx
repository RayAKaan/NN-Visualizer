import { useLabStore } from "../../../store/labStore";

export function TrainedVsUntrainedToggle() {
  const comparisonMode = useLabStore((s) => s.comparisonMode);
  const setComparisonMode = useLabStore((s) => s.setComparisonMode);
  const isRunning = useLabStore((s) => s.isRunning);

  const modes: Array<{ value: "trained" | "untrained"; label: string }> = [
    { value: "trained", label: "Trained" },
    { value: "untrained", label: "Untrained" },
  ];

  return (
    <div className="inline-flex items-center gap-2 rounded-xl border px-2 py-1"
      style={{ background: "var(--bg-panel)", borderColor: "var(--glass-border)" }}>
      <span className="text-xs" style={{ color: "var(--text-3)" }}>Weights</span>
      <div className="flex rounded-lg p-0.5" style={{ background: "var(--bg-void)" }}>
        {modes.map((mode) => (
          <button key={mode.value} type="button" disabled={isRunning}
            onClick={() => setComparisonMode(comparisonMode === mode.value ? "off" : mode.value)}
            className="rounded-md px-2 py-1 text-xs"
            style={{
              background: comparisonMode === mode.value ? (mode.value === "trained" ? "var(--fwd-bg)" : "var(--status-warning-bg)") : "transparent",
              color: comparisonMode === mode.value ? (mode.value === "trained" ? "var(--fwd)" : "var(--status-warning)") : "var(--text-4)",
            }}>
            {mode.label}
          </button>
        ))}
      </div>
    </div>
  );
}
