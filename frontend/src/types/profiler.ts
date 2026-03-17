export interface LayerProfile {
  stageId: string;
  flops: number;
  memoryBytes: number;
  parameterCount: number;
  parameterBytes: number;
  inferenceTimeMs: number;
  flopPercent: number;
  memoryPercent: number;
  paramPercent: number;
  timePercent: number;
  flopsPerParam: number;
  memoryEfficiency: number;
}

export interface NetworkProfile {
  architecture: string;
  totalFlops: number;
  totalMemoryBytes: number;
  totalParams: number;
  totalInferenceMs: number;
  layers: LayerProfile[];
  bottleneckLayer: string;
  bottleneckType: "compute" | "memory" | "parameters";
  comparisonProfiles?: Record<string, NetworkProfile>;
}
