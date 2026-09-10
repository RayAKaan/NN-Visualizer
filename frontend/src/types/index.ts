export type ModelType = "ann" | "cnn" | "rnn";

export type ModelFamily = "ANN" | "CNN" | "RNN";

export type RegistryStatus =
  | "available"
  | "loading"
  | "loaded"
  | "unavailable"
  | "error";

export interface CatalogModel {
  id: string;
  name: string;
  family: ModelFamily;
  framework?: string | null;
  source?: string | null;
  weights?: string | null;
  dataset?: string | null;
  input_type?: "mnist_pixels" | "image" | "text" | null;
  input_shape?: number[] | null;
  num_classes?: number | null;
  preprocessing?: string | null;
  architecture?: string | null;
  parameter_count?: number | null;
  description?: string | null;
  license?: string | null;
  pretrained?: boolean;
  status: RegistryStatus;
  unavailable_reason?: string | null;
  error?: string | null;
}

export interface LayerInfo {
  name: string;
  type: string;
  shape: number[];
  activations?: number[] | number; 
  feature_maps?: number[][][]; 
  weights?: number[];
  bias?: number[];
}

export interface PredictionResult {
  prediction: number;
  confidence: number;
  probabilities: number[];
  layers: LayerInfo[];
  model_type: string;
  explanation?: any;
}

export interface TrainingConfig {
  model_type: ModelType;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  optimizer: string;
}

export interface TrainingMetrics {
  epoch: number;
  total_epochs?: number;
  loss: number;
  accuracy: number;
  val_loss: number;
  val_accuracy: number;
  precision_per_class?: number[];
  recall_per_class?: number[];
  f1_per_class?: number[];
  confusion_matrix?: number[][];
}

export interface TrainingBatchMetrics {
  epoch: number;
  batch: number;
  total_batches: number;
  total_epochs?: number;
  model_type?: string;
  loss: number;
  accuracy: number;
  learning_rate: number;
  gradient_norm: number;
  timestamp?: number;
}

export interface TrainingStatus {
  status: "idle" | "training" | "paused" | "stopping" | "stopped" | "completed";
  current_epoch: number;
  total_epochs: number;
}
