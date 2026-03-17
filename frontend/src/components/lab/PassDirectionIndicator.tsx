import { useLabStore } from "../../store/labStore";

export function PassDirectionIndicator() {
  const passDirection = useLabStore((s) => s.passDirection);
  return (
    <div className="mb-3 inline-flex items-center gap-2 rounded-lg border px-3 py-1.5 text-xs"
      style={{
        borderColor: passDirection === "forward" ? "var(--fwd-border)" : "var(--bwd-border)",
        background: passDirection === "forward" ? "var(--fwd-bg)" : "var(--bwd-bg)",
        color: passDirection === "forward" ? "var(--fwd)" : "var(--bwd)",
      }}>
      <span>{passDirection === "forward" ? "?" : "?"}</span>
      <span>{passDirection === "forward" ? "Forward Pass" : "Backward Pass"}</span>
    </div>
  );
}
