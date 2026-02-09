export type BatchMessage = {
  type: "batch";
  epoch: number;
  batch: number;
  loss: number;
  accuracy: number;
  activations: {
    hidden1: number[];
    hidden2: number[];
    output: number[];
  };
  gradients: {
    hidden1_hidden2: number[][];
    hidden2_output: number[][];
  };
};

export type EpochMessage = {
  type: "epoch";
  epoch: number;
  loss: number;
  accuracy: number;
  val_loss: number;
  val_accuracy: number;
};

export type WeightsMessage = {
  type: "weights";
  epoch: number;
  weights: {
    hidden1_hidden2: number[][];
    hidden2_output: number[][];
  };
};

export type TrainingControlMessage = {
  command: string;
  [key: string]: unknown;
};

export type TrainingMessage = BatchMessage | EpochMessage | WeightsMessage | { type: "stopped" };
