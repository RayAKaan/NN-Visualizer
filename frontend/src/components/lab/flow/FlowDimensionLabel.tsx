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
    <div className={`mb-2 mt-1 text-center text-[12px] ${isActive ? "text-ink-soft" : "text-ink-faint"}`}>
      <div>{shape.join("x")}</div>
      <div>{(statistics.sparsity * 100).toFixed(0)}% sparse</div>
    </div>
  );
}
