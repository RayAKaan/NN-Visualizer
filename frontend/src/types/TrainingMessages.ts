export type BatchUpdateMessage = {
  type: "batch_update";
  epoch: number;
  batch: number;
  activations: {
    hidden1: number[];
    hidden2: number[];
    output: number[];
  };
  gradients: {
    hidden1_hidden2: number[][];
    hidden2_output: number[][];
  };
  weights: {
    hidden1_hidden2: number[][];
    hidden2_output: number[][];
  };
  loss: number;
  accuracy: number;
  learning_rate: number;
  gradient_norm: number;
  timestamp: number;
};

export type EpochUpdateMessage = {
  type: "epoch_update";
  epoch: number;
  loss: number;
  accuracy: number;
  val_loss: number;
  val_accuracy: number;
};

export type WeightsUpdateMessage = {
  type: "weights_update";
  epoch: number;
  weights: {
    hidden1_hidden2: number[][];
    hidden2_output: number[][];
  };
};

export type StatusMessage = {
  type: "status";
  status: string;
};

export type ErrorMessage = {
  type: "error";
  message: string;
};

export type TrainingMessage =
  | BatchUpdateMessage
  | EpochUpdateMessage
  | WeightsUpdateMessage
  | StatusMessage
  | ErrorMessage
  | { type: "training_stopped" }
  | { type: "training_complete"; metrics?: Record<string, unknown> };
