export type ModelType = "ann" | "cnn" | "rnn";
export type ViewMode = "2d" | "3d" | "compare";
export type AppMode = "predict" | "train";

export interface LayerInfo { name: string; type: string; output_shape: string; params: number; activation?: string | null }
export interface ModelInfo { model_type: ModelType; loaded: boolean; total_params?: number; layers?: LayerInfo[] }
export interface ModelsAvailable { available: ModelType[]; active: ModelType }

export interface PredictionResult { model_type: "ann"; prediction: number; confidence: number; probabilities: number[]; layers: { hidden1: number[]; hidden2: number[]; hidden3: number[] }; explanation?: ANNExplanation }
export interface FeatureMapLayer { layer_name: string; layer_type: string; shape: number[]; top_k: number; activation_ranking: number[]; feature_maps: number[][][]; mean_activations: number[] }
export interface KernelLayer { layer_name: string; kernel_shape: number[]; kernels: number[][][][] }
export interface CNNPredictionResult { model_type: "cnn"; prediction: number; confidence: number; probabilities: number[]; feature_maps: FeatureMapLayer[]; kernels: KernelLayer[]; dense_layers: Record<string, number[]>; explanation?: CNNExplanation }
export interface RNNPredictionResult { model_type: "rnn"; prediction: number; confidence: number; probabilities: number[]; timestep_activations: number[]; lstm_output: number[]; cell_state_summary: { mean: number; std: number; min: number; max: number }; dense_layers: Record<string, number[]>; explanation?: RNNExplanation }
export type AnyPredictionResult = PredictionResult | CNNPredictionResult | RNNPredictionResult;

export interface NeuronContribution { id: number; activation: number }
export interface ActiveFilter { layer: string; filter: number; mean: number }
export interface CompetitorDigit { digit: number; probability: number }
export interface ANNExplanation { model_type: "ann"; top_neurons: NeuronContribution[]; quadrant_evidence: string; competitors: CompetitorDigit[]; confidence_level: string; uncertainty_notes: string }
export interface CNNExplanation { model_type: "cnn"; active_filters: ActiveFilter[]; spatial_evidence: string; competitors: CompetitorDigit[]; confidence_level: string; uncertainty_notes: string }
export interface RNNExplanation { model_type: "rnn"; timestep_importance: number[]; sequential_summary: string; competitors: CompetitorDigit[]; confidence_level: string; uncertainty_notes: string }

export interface Edge { from: number; to: number; strength: number }
export interface NeuralState extends Partial<AnyPredictionResult> { edges?: Edge[] }

export type TrainingStatus = "idle" | "running" | "paused" | "stopping" | "completed" | "error";
export interface TrainingConfig { model_type: ModelType; learning_rate: number; batch_size: number; epochs: number; optimizer: string; activation: string; dropout_rate: number; kernel_initializer: string; lstm_units: number; bidirectional: boolean; conv1_filters: number; conv2_filters: number }
export interface LayerActivation { type: string; shape?: string; values?: number[]; means?: number[] }
export interface GradientInfo { norm: number; mean: number; std: number; max_abs: number; shape: string }
export interface WeightInfo { type: string; shape: string; mean: number; std: number; norm: number }
export interface BatchUpdate { type: "batch_update"; epoch: number; batch: number; total_batches: number; total_epochs: number; model_type: string; loss: number; accuracy: number; learning_rate: number; gradient_norm: number; activations: Record<string, LayerActivation>; gradients: Record<string, GradientInfo>; weights?: Record<string, WeightInfo> | null; timestamp: number }
export interface EpochUpdate { type: "epoch_update"; epoch: number; total_epochs: number; model_type: string; loss: number; accuracy: number; val_loss: number; val_accuracy: number; precision_per_class: number[]; recall_per_class: number[]; f1_per_class: number[]; confusion_matrix: number[][]; timestamp: number }
export type TrainingMessage = BatchUpdate | EpochUpdate | { type: "training_complete" } | { type: "training_stopped" } | { type: "training_error"; error: string };
export interface TrainingMetrics { losses: number[]; accuracies: number[]; gradientNorms: number[]; learningRates: number[]; valLosses: number[]; valAccuracies: number[]; epochBoundaries: number[]; precisionHistory: number[][]; recallHistory: number[][]; f1History: number[][]; confusionMatrices: number[][][] }
