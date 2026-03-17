export type Architecture = "ANN" | "CNN" | "RNN";
export type Dataset = "mnist" | "catdog";

export type StageStatus = "locked" | "active" | "processing" | "completed";
export type PassDirection = "forward" | "backward";

export type LayerType =
  | "input"
  | "preprocessing"
  | "dense"
  | "conv2d"
  | "activation_relu"
  | "max_pool"
  | "flatten"
  | "softmax"
  | "lstm_cell"
  | "output";

export interface EquationSet {
  primary: string;
  explanation?: string;
}

export interface BackwardEquationSet {
  chainRule: string;
  localGradient: string;
  weightGradient?: string;
  substituted?: string;
  result?: string;
  explanation: string;
}

export interface StageDefinition {
  id: string;
  name: string;
  type: LayerType;
  explanation: string;
  equations: EquationSet;
  inputShape: number[];
  outputShape: number[];
  params?: Record<string, number | string>;
}

export interface StageActivation {
  stageId: string;
  inputData: Float32Array;
  outputData: Float32Array;
  weights?: Float32Array;
  bias?: Float32Array;
  kernels?: Float32Array[];
  gateValues?: {
    forget: Float32Array;
    input: Float32Array;
    output: Float32Array;
    cellState: Float32Array;
  };
  metadata: {
    inputShape: number[];
    outputShape: number[];
    paramCount: number;
    computeTimeMs: number;
  };
}

export interface BackwardStageActivation {
  stageId: string;
  inputGradient: Float32Array;
  outputGradient: Float32Array;
  weightGradient?: Float32Array;
  biasGradient?: Float32Array;
  kernelGradients?: Float32Array[];
  gateGradients?: {
    forget: Float32Array;
    input: Float32Array;
    output: Float32Array;
    cellState: Float32Array;
  };
  stats: {
    inputGradMean: number;
    inputGradStd: number;
    inputGradMax: number;
    inputGradMin: number;
    weightGradMean?: number;
    weightGradStd?: number;
    weightGradMax?: number;
    gradientNorm: number;
    gradientFlowPercent: number;
    deadNeuronPercent?: number;
  };
  proposedWeightDelta?: Float32Array;
  proposedBiasDelta?: Float32Array;
  metadata: {
    inputShape: number[];
    outputShape: number[];
    computeTimeMs: number;
  };
}

export interface LossInfo {
  lossValue: number;
  lossType: "cross_entropy" | "mse" | "bce";
  perClassLoss: number[];
  trueLabel: number;
  predictedLabel: number;
  isCorrect: boolean;
  trueDistribution: number[];
  predictedDistribution: number[];
}

export interface SaliencyData {
  inputGradient: Float32Array;
  inputShape: number[];
  absoluteMax: number;
  topPixels: Array<{
    index: number;
    row: number;
    col: number;
    gradientValue: number;
    normalizedImportance: number;
  }>;
}

export interface WeightInspectionData {
  stageId: string;
  weights: Float32Array;
  bias: Float32Array | null;
  shape: number[];
  statistics: {
    mean: number;
    std: number;
    min: number;
    max: number;
    sparsity: number;
    distribution: {
      bins: number[];
      counts: number[];
    };
  };
  untrainedWeights?: Float32Array;
  untrainedStatistics?: WeightInspectionData["statistics"];
}

export interface WeightChange {
  index: number;
  row: number;
  col: number;
  originalValue: number;
  newValue: number;
}

export interface WeightManipulation {
  stageId: string;
  type: "single_weight" | "neuron_knockout" | "layer_bypass" | "bulk_randomize";
  changes: WeightChange[];
  originalPrediction: { label: number | string; confidence: number };
  manipulatedPrediction: { label: number | string; confidence: number } | null;
  impactScore: number;
}

export interface NeuronState {
  neuronIndex: number;
  isAlive: boolean;
  activationValue: number;
  outgoingWeightCount: number;
  importanceScore: number;
}

export interface ManipulationSnapshot {
  timestamp: number;
  stageId: string;
  changes: WeightChange[];
  resultingPrediction: { label: number | string; confidence: number };
}

export interface LayerEpochSnapshot {
  stageId: string;
  weights?: Float32Array;
  bias?: Float32Array;
  activationSample?: Float32Array;
  featureMapSample?: Float32Array[];
  statistics: {
    weightMean: number;
    weightStd: number;
    activationMean: number;
    activationSparsity: number;
    gradientNorm: number;
  };
}

export interface EpochCheckpoint {
  epoch: number;
  metrics: {
    trainLoss: number;
    valLoss: number;
    trainAccuracy: number;
    valAccuracy: number;
    learningRate: number;
  };
  layerSnapshots: Record<string, LayerEpochSnapshot>;
}

export interface PipelineState {
  architecture: Architecture;
  dataset: Dataset;
  stages: StageDefinition[];
  currentStageIndex: number;
  stageStatuses: Record<string, StageStatus>;
  activations: Record<string, StageActivation>;
  finalPrediction: {
    label: number | string;
    confidence: number;
    probs: number[];
  } | null;
  isRunning: boolean;
  speed: number;

  passDirection: PassDirection;
  trueLabel: number | null;
  lossInfo: LossInfo | null;

  backwardActivations: Record<string, BackwardStageActivation>;
  backwardStageStatuses: Record<string, StageStatus>;
  currentBackwardStageIndex: number;

  saliencyData: SaliencyData | null;

  comparisonMode: "off" | "trained" | "untrained";
  untrainedActivations: Record<string, StageActivation>;

  inspectedStageId: string | null;
  weightInspection: WeightInspectionData | null;

  learningRate: number;

  isManipulationMode: boolean;
  activeManipulations: Record<string, WeightManipulation>;
  manipulationHistory: ManipulationSnapshot[];
  bypassedLayers: Set<string>;
  knockedOutNeurons: Record<string, Set<number>>;
  modifiedWeights: Record<string, Map<number, number>>;
  manipulatedActivations: Record<string, StageActivation>;
  manipulatedPrediction: {
    label: number | string;
    confidence: number;
    probs: number[];
  } | null;
  isReInferencing: boolean;

  isReplayMode: boolean;
  availableEpochs: number[];
  currentReplayEpoch: number;
  epochCheckpoints: Record<number, EpochCheckpoint>;
  isLoadingEpoch: boolean;
  replayAutoPlaySpeed: number;
  isReplayAutoPlaying: boolean;
}
