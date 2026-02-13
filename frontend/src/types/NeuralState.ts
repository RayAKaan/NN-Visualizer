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
