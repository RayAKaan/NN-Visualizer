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
    <div className="inline-flex items-center gap-2 rounded-xl border border-barley-linestrong bg-white px-2 py-1">
      <span className="text-xs text-ink-mute">Weights</span>
      <div className="flex rounded-lg bg-barley-page p-0.5">
        {modes.map((mode) => (
          <button key={mode.value} type="button" disabled={isRunning}
            onClick={() => setComparisonMode(comparisonMode === mode.value ? "off" : mode.value)}
            className={`rounded-md px-2 py-1 text-xs ${comparisonMode === mode.value
              ? mode.value === "trained"
                ? "bg-arch-ann/10 text-arch-ann"
                : "bg-status-warning/10 text-status-warning"
              : "text-ink-faint"}`}>
            {mode.label}
          </button>
        ))}
      </div>
    </div>
  );
}
