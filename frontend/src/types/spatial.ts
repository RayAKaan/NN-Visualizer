export interface ReceptiveField {
  stageId: string;
  featureMapIndex: number;
  spatialPosition: [number, number];
  inputRegion: {
    top: number;
    left: number;
    bottom: number;
    right: number;
    width: number;
    height: number;
  };
  effectiveRegion: {
    centerRow: number;
    centerCol: number;
    effectiveRadius: number;
    attentionMap: Float32Array;
  };
}

export interface LayerAttention {
  stageId: string;
  attentionMap: Float32Array;
  topRegions: Array<{
    centerRow: number;
    centerCol: number;
    radius: number;
    importance: number;
    description: string;
  }>;
}
