export type LayerType =
  | "input"
  | "dense"
  | "output"
  | "conv2d"
  | "maxpool2d"
  | "avgpool2d"
  | "flatten"
  | "batchnorm"
  | "rnn"
  | "lstm"
  | "gru"
  | "embedding"
  | "attention"
  | "residual";

export interface LayerConfig {
  type: LayerType;
  neurons: number;
  activation?: string;
  init?: string;
  input_shape?: number[];
  kernel_size?: number;
  stride?: number;
  padding?: string;
  filters?: number;
  pool_size?: number;
  pool_stride?: number;
  hidden_size?: number;
  sequence_length?: number;
  return_sequences?: boolean;
  embedding_dim?: number;
  vocab_size?: number;
  num_heads?: number;
}

export interface ValidationResult {
  valid: boolean;
  architecture: number[];
  activations: string[];
  total_params: number;
  flops_per_sample: number;
  layer_params: Array<{ layer: number; weights: number; biases: number; total: number }>;
  errors: string[];
  warnings: string[];
}

export interface BuildResponse {
  graph_id: string;
  weights: number[][][];
  biases: number[][];
  weight_stats: Array<{ layer: number; mean: number; std: number; min: number; max: number }>;
}

export interface ForwardStep {
  step_index: number;
  layer_index: number;
  operation: "matmul" | "bias_add" | "activation" | "forward";
  description: string;
  equation_text?: string;
  activation_name?: string;
  input_values: number[];
  output_values: number[];
  weights_snapshot?: number[][];
  bias_snapshot?: number[];
}

export interface ForwardFullResponse {
  steps: ForwardStep[];
  final_output: number[];
  total_steps: number;
  layer_outputs: Record<string, number[]>;
}

export interface ForwardStepResponse {
  step: ForwardStep;
  completed_layers: number[];
  active_layer: number;
  partial_activations: Record<string, number[]>;
}

export interface EquationResponse {
  layer_index: number;
  layer_type: string;
  activation: string;
  generic_equations: {
    pre_activation: string;
    activation: string;
    per_neuron: string;
  };
  activation_plot: {
    name: string;
    formula: string;
    domain: [number, number];
    points: Array<[number, number]>;
  };
  dimensions: {
    W_shape: [number, number];
    b_shape: [number];
    input_dim: number;
    output_dim: number;
    param_count: number;
  };
  weight_stats: {
    mean: number;
    std: number;
    min: number;
    max: number;
  };
  numeric_equations: Array<{
    neuron_index: number;
    pre_activation_eq: string;
    activation_eq: string;
    pre_activation_value: number;
    activation_value: number;
    is_dead: boolean;
  }>;
}

export interface DatasetPoint {
  x: number[];
  y: number[];
}

export interface DatasetResponse {
  dataset_id: string;
  train: DatasetPoint[];
  test: DatasetPoint[];
  stats: {
    n_train: number;
    n_test: number;
    class_balance: Record<string, number>;
    feature_range: { x1: [number, number]; x2: [number, number] };
    feature_mean: number[];
    feature_std: number[];
  };
}

export interface WeightInspection {
  layer_index: number;
  weight_matrix: number[][];
  bias_vector: number[];
  shape: [number, number];
  stats: {
    mean: number;
    std: number;
    min: number;
    max: number;
    l2_norm: number;
    sparsity: number;
  };
  histogram: {
    bins: number[];
    counts: number[];
  };
}

export interface ActivationInspection {
  layer_index: number;
  pre_activation: number[];
  post_activation: number[];
  dead_neurons: number[];
  activation_name: string;
  histogram: {
    bins: number[];
    counts: number[];
  };
}

export interface BackwardStep {
  step_index: number;
  layer_index: number;
  operation: string;
  description: string;
  equation_text: string;
  equation_detailed?: string[];
  output_values?: number[];
  gradient_values?: number[][];
  delta_values?: number[];
}

export interface BackwardFullResponse {
  loss_value: number;
  loss_equation: string;
  steps: BackwardStep[];
  total_steps: number;
  gradients_W: number[][][];
  gradients_b: number[][];
  deltas: number[][];
  gradient_summary: {
    per_layer: Array<{
      layer: number;
      dW_norm: number;
      db_norm: number;
      dW_mean: number;
      dW_std: number;
      dW_min: number;
      dW_max: number;
      delta_norm: number;
    }>;
    total_gradient_norm: number;
    max_gradient: number;
    min_gradient: number;
  };
}

export interface TrainingConfig {
  epochs: number;
  batch_size: number;
  learning_rate: number;
  optimizer: string;
  loss_function: string;
  l2_lambda: number;
  dropout_rate: number;
  lr_scheduler?: string | null;
  lr_decay_rate?: number | null;
  lr_step_size?: number | null;
  shuffle?: boolean;
  snapshot_interval?: number;
}

export interface TrainingMetrics {
  train_loss: number;
  test_loss: number;
  train_accuracy: number;
  test_accuracy: number;
  learning_rate: number;
  gradient_norms: number[];
  weight_norms: number[];
  weight_deltas: number[];
  dead_neurons: number[];
  epoch_duration_ms: number;
}

export interface FeatureMapResponse {
  per_layer: Array<{
    layer_index: number;
    layer_type: string;
    n_filters: number;
    output_shape: number[];
    feature_maps: Array<{
      filter_index: number;
      map: number[][];
      map_base64: string;
      max_activation: number;
      mean_activation: number;
      sparsity: number;
    }>;
    filter_kernels: Array<{
      filter_index: number;
      kernel: number[][];
      kernel_base64: string;
      description: string;
    }>;
  }>;
}

export interface SaliencyResponse {
  method: string;
  saliency_map: number[][];
  saliency_base64: string;
  overlay_base64: string;
  top_pixels: Array<{ row: number; col: number; importance: number }>;
}

export interface FilterResponse {
  filter_index: number;
  kernel: number[][];
  kernel_description: string;
  top_activating_samples: Array<{
    sample_index: number;
    image_base64: string;
    max_activation: number;
    activation_map_base64: string;
  }>;
}

export interface NeuronAtlasResponse {
  layer_index: number;
  neurons: Array<{ index: number; mean: number; std: number; sparsity: number }>;
}

export interface SequenceStepResponse {
  timestep: number;
  input_t: number[];
  previous_hidden?: number[];
  previous_cell?: number[];
  new_hidden: number[];
  new_cell?: number[];
  gates?: Record<string, { values: number[]; equation: string }>;
  hidden_history: Array<{ t: number; h: number[] }>;
}

export interface SequenceFullResponse {
  all_hidden_states: number[][];
  all_cell_states: number[][];
  all_gate_values: any;
  final_output: number[] | number[][];
  all_attention_weights?: number[][];
  attention_heatmap_base64?: string;
}

export interface CompareSetupResponse {
  comparison_id: string;
  models: Array<{ model_id: string; graph_id: string | null; total_params: number }>;
  epochs: number;
  dataset_id: string;
}

export interface CompareResultsResponse {
  comparison_id: string;
  models: any[];
  winner: any;
  loss_histories: Record<string, any>;
}

export interface ProfileResponse {
  summary: Record<string, number>;
  per_layer: any[];
  batch_analysis: any[];
  bottleneck: any;
  memory: any;
}


export interface TemplateListResponse {
  templates: Array<{ id: string; name: string; category: string; layers: LayerConfig[] }>;
}

export interface ImportUploadResponse {
  import_id: string;
  status: string;
  format_detected: string;
  architecture: any;
  warnings: string[];
}

export interface ImportBuildResponse {
  graph_id: string;
  ready: boolean;
}

export interface ExportCodeResponse {
  format: string;
  code: string;
  filename: string;
}

export interface ExportImageResponse {
  image_base64: string;
  filename: string;
}

export interface LandscapeResult {
  grid_x: number[];
  grid_y: number[];
  loss_surface: number[][];
  center_point: number[];
  center_loss: number;
  min_loss: number;
  min_location: number[];
  max_loss: number;
  critical_points: any[];
  training_trajectory: any[];
  sharpness_score: number;
  flatness_description: string;
}

export interface LandscapeComputeResponse {
  task_id: string;
  status: string;
  total_evaluations: number;
  estimated_time_seconds: number;
}

export interface LandscapeStatusResponse {
  status: string;
  progress: number;
  result: LandscapeResult | null;
}

export interface EmbeddingProjection {
  index: number;
  coords: number[];
  label: number;
  predicted: number;
}

export interface EmbeddingResponse {
  projections: EmbeddingProjection[];
  variance_explained: number[];
  cluster_metrics: {
    silhouette_score: number;
    inter_class_distance: number;
    intra_class_distance: number;
  };
  method_params_used: Record<string, any>;
}

export interface IntegratedGradientsResponse {
  attributions: number[];
  attributions_base64: string;
  convergence_delta: number;
  baseline_output: number;
  input_output: number;
  attribution_sum: number;
  top_features: Array<{ index: number[]; attribution: number }>;
}
export interface InterpretStubResponse {
  status: "not_implemented";
  message: string;
}
export interface AdversarialAttackResponse {
  adversarial_input: number[];
  perturbation: number[];
  adversarial_image_base64: string;
  perturbation_amplified_base64: string;
  original_prediction: { class: number; confidence: number };
  adversarial_prediction: { class: number; confidence: number };
  attack_success: boolean;
  perturbation_stats: { linf_norm: number; l2_norm: number; l0_norm: number; mean_perturbation: number };
  confidence_shift: Array<{ class: number; original: number; adversarial: number }>;
}

export interface RobustnessCurveResponse {
  robustness_curve: Array<{ epsilon: number; accuracy: number; mean_confidence: number }>;
}

export interface PruneResponse {
  original_params: number;
  remaining_params: number;
  sparsity_achieved: number;
  per_layer_sparsity: Array<{ layer: number; sparsity: number; remaining: number; total: number }>;
  pruning_mask: number[][][];
}

export interface QuantizeResponse {
  original_memory_bytes: number;
  quantized_memory_bytes: number;
  compression_ratio: number;
  per_layer_quantization_error: Array<{ layer: number; mean_error: number; max_error: number }>;
}

export interface PruneSweepResponse {
  sweep_results: Array<{ sparsity: number; accuracy: number; params: number; flops: number }>;
  recommended_sparsity: number;
  pareto_frontier: number[];
}
export interface GenerativeSampleResponse {
  mode: string;
  samples: string[];
}

export interface AugmentationPreviewResponse {
  samples: string[];
}
export interface ExperimentRecord {
  id: string;
  name: string;
  config: Record<string, any>;
  metrics?: Record<string, any> | null;
  notes?: string | null;
  tags?: string[] | null;
}

export interface ExperimentsListResponse {
  experiments: ExperimentRecord[];
}

export interface AssistantResponse {
  reply: string;
  actions: any[];
}
export interface ShapResponse {
  shap_values: number[];
  shap_base64: string;
  base_value: number;
  output_value: number;
}

export interface LimeResponse {
  superpixel_map: number[][];
  superpixel_importances: Array<{ segment_id: number; importance: number; sign: string }>;
  explanation_image_base64: string;
  local_model_r2: number;
  local_model_weights: number[];
}

export interface LrpResponse {
  relevance_per_layer: Array<{ layer: number; relevance: number[]; total_relevance: number }>;
  input_relevance: number[];
  input_relevance_base64: string;
  conservation_error: number;
}

