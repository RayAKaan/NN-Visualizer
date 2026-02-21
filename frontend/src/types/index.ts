export type ModelType = "ann" | "cnn";

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

export type FeatureMapLayer = {
  layer_name: string;
  layer_type: "conv" | "pool";
  shape: [number, number, number];
  top_k: number;
  activation_ranking: number[];
  feature_maps: Record<string, number[][]>;
  mean_activations: number[];
};

export type KernelLayer = {
  layer_name: string;
  kernel_shape: number[];
  kernels: Record<string, number[][]>;
};

export type CNNExplanationResult = {
  model_type: "cnn";
  prediction: number;
  confidence: number;
  confidence_level: string;
  active_filters: {
    layer: string;
    filter_index: number;
    mean_activation: number;
  }[];
  spatial_evidence: string[];
  competitors: CompetitorDigit[];
  uncertainty_notes: string[];
};

export type PredictionResult = {
  model_type?: "ann";
  prediction: number;
  probabilities: number[];
  layers: {
    hidden1: number[];
    hidden2: number[];
  };
  explanation: ExplanationResult;
};

export type CNNPredictionResult = {
  model_type: "cnn";
  prediction: number;
  confidence: number;
  probabilities: number[];
  feature_maps: FeatureMapLayer[];
  kernels: KernelLayer[];
  dense_layers: Record<string, number[]>;
  layer_activations: Record<
    string,
    {
      type: "spatial" | "dense";
      shape?: number[];
      global_mean?: number;
      values?: number[];
    }
  >;
  explanation: CNNExplanationResult;
};

export type AnyPredictionResult = PredictionResult | CNNPredictionResult;

export type LayerInfo = {
  name: string;
  size?: number;
  activation?: string;
  type?: string;
  output_shape?: string;
  params?: number;
  filters?: number;
  kernel_size?: [number, number] | number[];
  pool_size?: [number, number] | number[];
  units?: number;
  rate?: number;
};

export type ModelInfo = {
  type: string;
  layers: LayerInfo[];
  total_params: number;
  accuracy?: number;
  input_shape?: string;
  output_shape?: string;
};

export type ModelsAvailable = {
  available: ModelType[];
  active: ModelType;
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
    hidden1_hidden2?: Edge[];
    hidden2_output?: Edge[];
    [key: string]: Edge[] | undefined;
  };
};
