export interface EdgeState {
  id: string;
  from: string;
  to: string;
  weight: number;
  gradient: number;
  contribution: number;
}

export interface NeuronState {
  id: string;
  layerType: "input" | "dense" | "conv" | "pool" | "recurrent";
  activation: number;
  bias: number;
  gradient: number;
  incomingEdges: EdgeState[];
  outgoingEdges: EdgeState[];
}

export type IntrospectionMode = "prediction" | "training";
export type ArchitectureType = "ann" | "cnn" | "rnn";
