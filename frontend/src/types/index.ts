export type NeuronContribution = {
  layer: string;
  index: number;
  activation: number;
  contribution: number;
};

export type CompetitorDigit = {
  digit: number;
  probability: number;
  reason: string;
};

export type ExplanationResult = {
  prediction: number;
  confidence: number;
  confidence_level: string;
  top_neurons: NeuronContribution[];
  evidence: string[];
  competitors: CompetitorDigit[];
  uncertainty_notes: string[];
};

export type PredictionResult = {
  prediction: number;
  probabilities: number[];
  layers: {
    hidden1: number[];
    hidden2: number[];
  };
  explanation: ExplanationResult;
};

export type LayerInfo = {
  name: string;
  size: number;
  activation?: string;
};

export type ModelInfo = {
  type: string;
  layers: LayerInfo[];
  total_params: number;
  accuracy: number;
};

export type Edge = {
  from: number;
  to: number;
  strength: number;
};

export type NeuralState = {
  input: number[];
  layers: {
    hidden1: number[];
    hidden2: number[];
    output: number[];
  };
  prediction: number;
  confidence: number;
  edges: {
    hidden1_hidden2: Edge[];
    hidden2_output: Edge[];
  };
};
