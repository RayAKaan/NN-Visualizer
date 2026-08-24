import { useLabStore } from "../../store/labStore";

export function PassDirectionIndicator() {
  const passDirection = useLabStore((s) => s.passDirection);
  return (
    <div
      className={`mb-3 inline-flex items-center gap-2 rounded-lg border px-3 py-1.5 text-xs ${
        passDirection === "forward"
          ? "border-arch-ann/25 bg-arch-ann/[0.07] text-arch-ann"
          : "border-ember-700/20 bg-ember-700/[0.06] text-ember-800"
      }`}
    >
      <span>{passDirection === "forward" ? "\u2192" : "\u2190"}</span>
      <span>{passDirection === "forward" ? "Forward Pass" : "Backward Pass"}</span>
    </div>
  );
}
