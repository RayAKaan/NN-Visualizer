export interface FlowSnapshot {
  stageId: string;
  stageIndex: number;
  shape: number[];
  dimensionality: "1d" | "2d" | "3d";
  thumbnail: {
    url: string;
    width: number;
    height: number;
  };
  statistics: {
    mean: number;
    std: number;
    min: number;
    max: number;
    sparsity: number;
    entropy: number;
    dimensionalReduction: number;
  };
  dominantPattern: string;
}

export interface FlowTransition {
  fromStageId: string;
  toStageId: string;
  transformationType: string;
  informationRetained: number;
  dimensionChange: {
    before: number[];
    after: number[];
    elementCountBefore: number;
    elementCountAfter: number;
    compressionRatio: number;
  };
}
