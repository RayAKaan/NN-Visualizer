export interface NeuronIdentity {
  stageId: string;
  neuronIndex: number;
  layerType: string;
  incomingConnections: number;
  outgoingConnections: number;
  strongestIncoming: Array<{
    fromNeuronIndex: number;
    fromStageId: string;
    weight: number;
  }>;
  strongestOutgoing: Array<{
    toNeuronIndex: number;
    toStageId: string;
    weight: number;
  }>;
  currentActivation: number;
  currentGradient: number | null;
  isAlive: boolean;
  featureVisualization?: string;
  topActivatingInputs?: Array<{
    inputIndex: number;
    activation: number;
    thumbnail: string;
  }>;
  importanceScore: number;
  ablationImpact: number;
  redundancyScore: number;
  activationHistory?: Array<{
    epoch: number;
    meanActivation: number;
    activationFrequency: number;
  }>;
  weightHistory?: Array<{
    epoch: number;
    incomingWeightMean: number;
    incomingWeightStd: number;
  }>;
}
