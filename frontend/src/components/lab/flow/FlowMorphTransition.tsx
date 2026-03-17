import type { FlowSnapshot } from "../../../types/flow";

interface Props {
  from: FlowSnapshot;
  to: FlowSnapshot;
  isActive: boolean;
}

export function FlowMorphTransition({ from, to, isActive }: Props) {
  const fromSize = Math.max(1, from.shape.reduce((a, b) => a * b, 1));
  const toSize = Math.max(1, to.shape.reduce((a, b) => a * b, 1));
  const ratio = toSize / fromSize;
  const compressing = ratio < 0.9;
  const expanding = ratio > 1.1;

  return (
    <div className="relative my-1 flex h-7 items-center justify-center">
      <div
        className="h-6 rounded-full"
        style={{
          width: compressing ? 8 : expanding ? 20 : 12,
          background: isActive ? "var(--fwd)" : "var(--text-4)",
          opacity: isActive ? 0.35 : 0.15,
          transition: "all .2s ease",
        }}
      />
      {(compressing || expanding) && (
        <span
          className="absolute -right-5 text-[8px] font-mono"
          style={{ color: compressing ? "var(--warning)" : "var(--info)" }}
        >
          {compressing ? "-" : "+"}
          {Math.abs((ratio - 1) * 100).toFixed(0)}%
        </span>
      )}
    </div>
  );
}
