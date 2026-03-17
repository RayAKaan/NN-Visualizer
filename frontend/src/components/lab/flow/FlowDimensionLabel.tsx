interface Props {
  shape: number[];
  statistics: {
    sparsity: number;
    dimensionalReduction: number;
  };
  isActive: boolean;
}

export function FlowDimensionLabel({ shape, statistics, isActive }: Props) {
  return (
    <div className="mb-2 mt-1 text-center text-[8px]" style={{ color: isActive ? "var(--text-2)" : "var(--text-4)" }}>
      <div>{shape.join("x")}</div>
      <div>{(statistics.sparsity * 100).toFixed(0)}% sparse</div>
    </div>
  );
}
