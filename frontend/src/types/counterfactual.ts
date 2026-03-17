export interface CounterfactualExperiment {
  id: string;
  timestamp: number;
  originalInput: Float32Array;
  modifiedInput: Float32Array;
  perturbationMask: Float32Array;
  originalPrediction: {
    label: number | string;
    confidence: number;
    probs: number[];
  };
  modifiedPrediction: {
    label: number | string;
    confidence: number;
    probs: number[];
  };
  predictionFlipped: boolean;
  confidenceChange: number;
  perturbationMagnitude: number;
  affectedPixelCount: number;
}

export interface SensitivityMapData {
  perPixelSensitivity: Float32Array;
  topSensitivePixels: Array<{
    index: number;
    row: number;
    col: number;
    sensitivity: number;
    direction: "increase" | "decrease";
  }>;
  overallSensitivity: number;
}
