export type ExecutionCursorState = "idle" | "running" | "computing" | "completed" | "error";
export type LabViewMode = "execution" | "classic";

export type Operation =
  | "input"
  | "dense"
  | "conv2d"
  | "max_pool"
  | "avg_pool"
  | "flatten"
  | "activation"
  | "lstm"
  | "lstm_bidirectional"
  | "dropout"
  | "softmax"
  | string;

export interface NetworkLayerDef {
  layerId: string;
  name: string;
  type: string;
  operation: Operation;
  activation: string | null;
  recurrentActivation?: string | null;
  inputShape: number[];
  outputShape: number[];
  params: number;
  trainable: boolean;
}

export interface NetworkMeta {
  id: "ann" | "cnn" | "rnn";
  name: string;
  framework: string;
  task: string;
  numClasses: number;
  inputShape: number[];
  outputShape: number[];
  totalParams: number;
  trained: boolean;
  layers: NetworkLayerDef[];
}

export interface TraceStageStatistics {
  mean: number;
  std: number;
  min: number;
  max: number;
  absMax: number;
  sparsity: number;
  positive: number;
  negative: number;
  zero: number;
}

export interface TraceStage {
  stageId: string;
  layerId: string;
  name: string;
  type: string;
  operation: Operation;
  activation: string | null;
  inputShape: number[];
  outputShape: number[];
  params: number;
  compute_time_ms: number;
  timing_method: string;
  statistics: TraceStageStatistics;
  feature_maps?: number[][][];
  output_data?: number[];
  input_data?: number[];
  sampled?: boolean;
  weights_kernel_shape?: number[] | null;
  kernels?: { kernel: Array<number[][][]>; bias: number[] };
  bias?: number[];
}

export interface ExecutionTrace {
  execution_id: string;
  model: NetworkMeta;
  input: {
    dataset: string;
    source: string;
    shape: number[];
    values: number[];
  };
  stages: TraceStage[];
  prediction: {
    label: number | string;
    labelText: string;
    confidence: number;
    probabilities: number[];
  };
  timing: {
    totalMs: number;
    totalMethod: string;
    perLayerMs: Record<string, number>;
    perLayerMethod: string;
  };
  capabilities: {
    neuronDetail: boolean;
    convDetail: boolean;
    lstmDetail: boolean;
    hasDense: boolean;
    hasConv: boolean;
    hasLstm: boolean;
    hasBackward: boolean;
  };
  meta: { trained: boolean; executedAt: number };
}

export interface DenseNeuronDetail {
  layerId: string;
  neuronIndex: number;
  units: number;
  activation: string | null;
  preActivation: number;
  bias: number;
  postActivation: number;
  matchedOutput: boolean;
  inputLength: number;
  contributionCount: number;
  maxAbsContribution: number;
  topContributions: Array<{
    index: number;
    inputValue: number;
    weight: number;
    contribution: number;
  }>;
  weightRowStats: TraceStageStatistics & { norm: number };
}

export interface ConvCellDetail {
  layerId: string;
  filterIndex: number;
  position: { h: number; w: number };
  patch: number[][][];
  kernel: number[][];
  bias: number;
  preActivation: number;
  activation: string | null;
  postActivation: number;
  actualOutput: number;
  matchedOutput: boolean;
  stride: number[];
  padding: string;
  inputShape: number[];
  outputShape: number[];
  featureMaps?: number[][][];
  selectedChannel?: number;
}

export interface LstmDetailSummary {
  hiddenNorm: number[];
  cellNorm: number[];
  forgetMean: number[];
  inputMean: number[];
  outputMean: number[];
  cellMean: number[];
}

export interface LstmDetail {
  layerId: string;
  units: number;
  timesteps: number;
  verified: boolean;
  maxDeviation: number;
  summary: LstmDetailSummary;
  detailed: null | {
    timestep: number;
    hiddenState: number[];
    cellState: number[];
    gates: {
      input: number[];
      forget: number[];
      candidate: number[];
      output: number[];
      cell: number[];
    };
  };
}

export interface SelectionState {
  neuronIndex: number;
  filterIndex: number;
  position: { h: number; w: number } | null;
  timestep: number;
}